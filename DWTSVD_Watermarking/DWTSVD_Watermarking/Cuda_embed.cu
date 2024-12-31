#include <iostream>
#include "./cuda_embed.cuh"

#include <device_launch_parameters.h>
#include <math.h> 
#include <curand_kernel.h>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda.h>

#define PI 355.0 / 113.0

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val == 0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// CUDA kernel for Gaussian Blur
__global__ void gaussianBlurKernel(const unsigned char* input, double* diff_accum,
    int rows, int cols, const double* sigma_values, int num_sigmas) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int sigma_idx = blockIdx.z;

    if (sigma_idx < num_sigmas && row < rows && col < cols) {
        double sigma = sigma_values[sigma_idx];
        int ksize = (int)(ceil((20.0 / 3.0) * sigma - 1.0 / 3.0)); // Kernel size based on sigma
        ksize = (ksize % 2 == 0) ? ksize + 1 : ksize;
        int half_ksize = ksize / 2;

        double sum = 0.0, weight = 0.0;
        for (int ky = -half_ksize; ky <= half_ksize; ++ky) {
            for (int kx = -half_ksize; kx <= half_ksize; ++kx) {
                int x = min(max(col + kx, 0), cols - 1);
                int y = min(max(row + ky, 0), rows - 1);

                double dist2 = kx * kx + ky * ky;
                double gaussian_weight = exp(-dist2 / (2.0 * sigma * sigma)) / (2 * PI * sigma * sigma);
                sum += input[y * cols + x] * gaussian_weight;
                weight += gaussian_weight;
            }
        }

        double blurred = sum / weight;
        double diff = fabs(static_cast<double>(input[row * cols + col]) - blurred);
        atomicAddDouble(&diff_accum[row * cols + col], diff);
    }
}

// CUDA kernel for Median Filtering
__global__ void medianFilterKernel(const unsigned char* input, double* diff_accum,
    int rows, int cols, const int* kernel_sizes, int num_kernels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_idx = blockIdx.z;

    if (kernel_idx < num_kernels && row < rows && col < cols) {
        int ksize = kernel_sizes[kernel_idx];
        int half_ksize = ksize / 2;

        unsigned char window[121]; // Supports up to an 11x11 kernel
        int count = 0;

        for (int ky = -half_ksize; ky <= half_ksize; ++ky) {
            for (int kx = -half_ksize; kx <= half_ksize; ++kx) {
                int x = min(max(col + kx, 0), cols - 1);
                int y = min(max(row + ky, 0), rows - 1);
                window[count++] = input[y * cols + x];
            }
        }

        // Sort window to find the median
        for (int i = 0; i < count - 1; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (window[i] > window[j]) {
                    unsigned char temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        unsigned char median_value = window[count / 2];
        double diff = fabs(static_cast<double>(input[row * cols + col]) - median_value);
        atomicAdd(&diff_accum[row * cols + col], diff);
    }
}


// CUDA kernel for Additive White Gaussian Noise
__global__ void addNoiseKernel(unsigned char* input, double* diff_accum,
    int rows, int cols, const double* stddev_values, int num_stddevs, unsigned long long seed) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int stddev_idx = blockIdx.z;
    curandState state;
    curand_init(seed, 0, 0, &state);

    if (stddev_idx < num_stddevs && row < rows && col < cols) {
        double stddev = stddev_values[stddev_idx];
        double noise = curand_normal_double(&state) * stddev + 0; //opencv randn uses 0 as the mean by default
        double original_value = static_cast<double>(input[row * cols + col]);
        double noisy_value = max(0.0, min(255.0, original_value + noise));

        double diff = fabs(original_value - noisy_value);
        atomicAddDouble(&diff_accum[row * cols + col], diff);
    }
}

__device__ void gaussianBlurPixel(const uchar* input, double* output, int width, int height, double sigma) {
    int kernelSize = (int)(ceil((20.0 / 3.0) * sigma - 1.0 / 3.0));
    kernelSize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;
    int radius = kernelSize / 2;

    // Calculate Gaussian kernel weights on-the-fly
    double kernel[49]; // Max kernel size for radius=3
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            double weight = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
            kernel[(i + radius) * kernelSize + (j + radius)] = weight;
            sum += weight;
        }
    }
    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        kernel[i] /= sum; // Normalize
    }

    // Get pixel coordinates
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    double blurredValue = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            int nx = min(max(x + i, 0), width - 1);
            int ny = min(max(y + j, 0), height - 1);
            blurredValue += input[ny * width + nx] * kernel[(i + radius) * kernelSize + (j + radius)];
        }
    }
    output[y * width + x] = blurredValue;
}

__global__ void sharpenKernel(
    const uchar* input, double* blank_image, int width, int height,
    const double* sigma_values, const double* alpha_values,
    int num_sigmas, int num_alphas) {

    // Get coordinates in the 3D grid
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = blockIdx.z;

    if (x >= width || y >= height || z >= num_sigmas * num_alphas) return;

    // Determine sigma and alpha based on the z-dimension
    int sigmaIdx = z / num_alphas;
    int alphaIdx = z % num_alphas;

    double sigma = sigma_values[sigmaIdx];
    double alpha = alpha_values[alphaIdx];

    // Allocate shared memory for blurred image
    extern __shared__ double shared_blurred[];

    // Calculate Gaussian blur
    gaussianBlurPixel(input, shared_blurred, width, height, sigma);
    __syncthreads();

    // Apply sharpening and compute difference
    int idx = y * width + x;
    if (idx < width * height) {
        double originalPixel = input[idx];
        double blurredPixel = shared_blurred[idx];
        double sharpened = originalPixel + alpha * (originalPixel - blurredPixel);
        sharpened = fmax(0.0, fmin(255.0, sharpened)); // Clamp between 0 and 255
        double diff = fabs(sharpened - originalPixel);
        atomicAddDouble(&blank_image[idx], diff);
    }
}


// Bilinear interpolation for unsigned char
__device__ double bilinearInterpolate(const unsigned char* input, int width, int height, double x, double y) {
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = min(x1 + 1, width - 1);
    int y2 = min(y1 + 1, height - 1);

    // Interpolation weights
    double wx = x - x1;
    double wy = y - y1;

    // Interpolate
    double top = (1.0 - wx) * input[y1 * width + x1] + wx * input[y1 * width + x2];
    double bottom = (1.0 - wx) * input[y2 * width + x1] + wx * input[y2 * width + x2];
    return (1.0 - wy) * top + wy * bottom;
}

// Kernel for resizing and calculating differences
__global__ void resizeKernel(
    const unsigned char* input, double* blank_image,
    int original_width, int original_height,
    unsigned char* resized_down, unsigned char* resized_up,
    int resized_width, int resized_height,
    const double* scale_factors, int num_scales) {

    // Pixel coordinates
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = blockIdx.z; // Scale index

    if (x >= original_width || y >= original_height || z >= num_scales) return;

    double scale = scale_factors[z];

    // Resize down
    double src_x = x / scale;
    double src_y = y / scale;
    if (x < resized_width && y < resized_height) {
        resized_down[y * resized_width + x] = bilinearInterpolate(input, original_width, original_height, src_x, src_y);
    }

    // Resize up
    if (x < original_width && y < original_height) {
        double upscaled_x = x * scale;
        double upscaled_y = y * scale;
        resized_up[y * original_width + x] = bilinearInterpolate(resized_down, resized_width, resized_height, upscaled_x, upscaled_y);
    }

    // Difference calculation
    if (x < original_width && y < original_height) {
        double diff = abs(resized_up[y * original_width + x] - input[y * original_width + x]);
        atomicAddDouble(&blank_image[y * original_width + x], diff);
    }
}


// Main function to handle all operations
void processAllOperations(const cv::Mat& input, cv::Mat& diff_accum,
    const std::vector<double>& blur_sigmas, const std::vector<int>& median_kernels,
    const std::vector<double>& noise_stddevs, const std::vector<double>& sharpen_sigma_values,
    const std::vector<double>& sharpen_alpha_values, const std::vector<double>& resize_scale_values)
{
    
    // Common sizes and memory allocation
    int rows = input.rows;
    int cols = input.cols;
    size_t img_size = rows * cols * sizeof(unsigned char);
    size_t diff_accum_size = rows * cols * sizeof(double);

    // Allocate device memory
    unsigned char* d_input;
    double* d_diff_accum;
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_diff_accum, diff_accum_size);

    // Copy input image to device
    cudaMemcpy(d_input, input.data, img_size, cudaMemcpyHostToDevice);
    cudaMemset(d_diff_accum, 0, diff_accum_size);

    // 1. Process Gaussian Blur
    double* d_blur_sigmas;
    cudaMalloc((void**)&d_blur_sigmas, blur_sigmas.size() * sizeof(double));
    cudaMemcpy(d_blur_sigmas, blur_sigmas.data(), blur_sigmas.size() * sizeof(double), cudaMemcpyHostToDevice);
    dim3 threads(32, 32);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y, blur_sigmas.size());
    gaussianBlurKernel << <blocks, threads >> > (d_input, d_diff_accum, rows, cols, d_blur_sigmas, blur_sigmas.size());
    cudaFree(d_blur_sigmas);

    // 2. Process Median Filtering
    int* d_median_kernels;
    cudaMalloc((void**)&d_median_kernels, median_kernels.size() * sizeof(int));
    cudaMemcpy(d_median_kernels, median_kernels.data(), median_kernels.size() * sizeof(int), cudaMemcpyHostToDevice);

    blocks.z = median_kernels.size();
    medianFilterKernel << <blocks, threads >> > (d_input, d_diff_accum, rows, cols, d_median_kernels, median_kernels.size());
    cudaFree(d_median_kernels);

    // 3. Process Additive White Gaussian Noise
    double* d_noise_stddevs;
    cudaMalloc((void**)&d_noise_stddevs, noise_stddevs.size() * sizeof(double));
    cudaMemcpy(d_noise_stddevs, noise_stddevs.data(), noise_stddevs.size() * sizeof(double), cudaMemcpyHostToDevice);
    threads.z = noise_stddevs.size();
    blocks.z = noise_stddevs.size();
    addNoiseKernel << <blocks, threads >> > (d_input, d_diff_accum, rows, cols, d_noise_stddevs, noise_stddevs.size(), (long long int)time(NULL));
    cudaFree(d_noise_stddevs);

    // 4. Sharperning
    double* d_sharpen_sigmas, *d_sharpen_alphas;
    cudaMalloc((void**)&d_sharpen_sigmas, sharpen_sigma_values.size() * sizeof(double));
    cudaMalloc((void**)&d_sharpen_alphas, sharpen_alpha_values.size() * sizeof(double));
    threads.z = sharpen_alpha_values.size();
    blocks.z = sharpen_sigma_values.size();
    sharpenKernel << <blocks, threads >> > ( d_input, d_diff_accum, rows, cols, d_sharpen_sigmas, d_sharpen_alphas,sharpen_sigma_values.size(), sharpen_alpha_values.size());
    cudaFree(d_sharpen_sigmas);
    cudaFree(d_sharpen_alphas);

    // 5. Resizing using INTER_LINEAR (bilinear interpolation)
    double* d_scale;
    unsigned char * d_resized_down, * d_resized_up;
    cudaMalloc((void**)&d_scale, resize_scale_values.size() * sizeof(double));
    cudaMalloc((void**)&d_resized_down, img_size);
    cudaMalloc((void**)&d_resized_up, img_size);
    threads.z = resize_scale_values.size();
    blocks.z = resize_scale_values.size();
    resizeKernel << <blocks, threads >> > (d_input, d_diff_accum,  rows, cols, d_resized_down, d_resized_up, rows, cols, d_scale, resize_scale_values.size());
    cudaFree(d_scale);
    cudaFree(d_resized_down);
    cudaFree(d_resized_up);

    // Copy accumulated differences back to host
    cudaMemcpy(diff_accum.data, d_diff_accum, diff_accum_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_diff_accum);
}