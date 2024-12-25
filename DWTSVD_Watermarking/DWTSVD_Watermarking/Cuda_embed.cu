#include <iostream>
#include "./cuda_embed.cuh"

#include <device_launch_parameters.h>
#include <math.h> 
#include <curand_kernel.h>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda.h>

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
        int ksize = max(3, 2 * static_cast<int>(3 * sigma) + 1); // Kernel size based on sigma
        int half_ksize = ksize / 2;

        double sum = 0.0, weight = 0.0;
        for (int ky = -half_ksize; ky <= half_ksize; ++ky) {
            for (int kx = -half_ksize; kx <= half_ksize; ++kx) {
                int x = min(max(col + kx, 0), cols - 1);
                int y = min(max(row + ky, 0), rows - 1);

                double dist2 = kx * kx + ky * ky;
                double gaussian_weight = exp(-dist2 / (2.0 * sigma * sigma));
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
        atomicAddDouble(&diff_accum[row * cols + col], diff);
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
        double noise = stddev * curand_uniform_double(&state) * 2.0 - stddev; // Simplistic noise generator
        double original_value = static_cast<double>(input[row * cols + col]);
        double noisy_value = max(0.0, min(255.0, original_value + noise));

        double diff = fabs(original_value - noisy_value);
        atomicAddDouble(&diff_accum[row * cols + col], diff);
    }
}

// Main function to handle all operations
void processAllOperations(const cv::Mat& input, cv::Mat& diff_accum,
    const std::vector<double>& blur_sigmas,
    const std::vector<int>& median_kernels,
    const std::vector<double>& noise_stddevs) {
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

    // Process Gaussian Blur
    double* d_blur_sigmas;
    cudaMalloc((void**)&d_blur_sigmas, blur_sigmas.size() * sizeof(double));
    cudaMemcpy(d_blur_sigmas, blur_sigmas.data(), blur_sigmas.size() * sizeof(double), cudaMemcpyHostToDevice);
    dim3 threads(16, 16, blur_sigmas.size());
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y, blur_sigmas.size());
    gaussianBlurKernel << <blocks, threads >> > (d_input, d_diff_accum, rows, cols, d_blur_sigmas, blur_sigmas.size());
    cudaFree(d_blur_sigmas);

    // Process Median Filtering
    int* d_median_kernels;
    cudaMalloc((void**)&d_median_kernels, median_kernels.size() * sizeof(int));
    cudaMemcpy(d_median_kernels, median_kernels.data(), median_kernels.size() * sizeof(int), cudaMemcpyHostToDevice);
    threads.z = median_kernels.size();
    blocks.z = median_kernels.size();
    medianFilterKernel << <blocks, threads >> > (d_input, d_diff_accum, rows, cols, d_median_kernels, median_kernels.size());
    cudaFree(d_median_kernels);

    // Process Additive White Gaussian Noise
    double* d_noise_stddevs;
    cudaMalloc((void**)&d_noise_stddevs, noise_stddevs.size() * sizeof(double));
    cudaMemcpy(d_noise_stddevs, noise_stddevs.data(), noise_stddevs.size() * sizeof(double), cudaMemcpyHostToDevice);
    threads.z = noise_stddevs.size();
    blocks.z = noise_stddevs.size();
    addNoiseKernel << <blocks, threads >> > (d_input, d_diff_accum, rows, cols, d_noise_stddevs, noise_stddevs.size(), (long long int)time(NULL));
    cudaFree(d_noise_stddevs);

    // Copy accumulated differences back to host
    cudaMemcpy(diff_accum.data, d_diff_accum, diff_accum_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_diff_accum);
}