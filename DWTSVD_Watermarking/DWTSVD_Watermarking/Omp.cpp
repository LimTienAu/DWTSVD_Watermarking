
#include "ompwatermark.h"
#define PI (355.0 / 113.0)

vector<vector<double>> omp_createGaussianKernel(int kernelSize, double sigma) {
    vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize));
    double sum = 0.0;
    int halfSize = kernelSize / 2;

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            kernel[i + halfSize][j + halfSize] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
            sum += kernel[i + halfSize][j + halfSize];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

// Function to apply Gaussian blur
Mat omp_apply_blur(const Mat& img, double sigma) {
    // Ensure the input image is single-channel (grayscale)
    CV_Assert(img.channels() == 1);

    const double MAX_SIGMA = 10.0;
    if (sigma > MAX_SIGMA) {
        sigma = MAX_SIGMA;
    }

    // Determine the kernel size based on sigma (6*sigma + 1)
    int kernelSize = static_cast<int>(ceil(6 * sigma)) | 1; // Ensure kernel size is odd
    int halfSize = kernelSize / 2;

    // Create Gaussian kernel
    vector<vector<double>> kernel = omp_createGaussianKernel(kernelSize, sigma);

    // Create output Mat
    Mat blurred = Mat::zeros(img.size(), img.type());

    int rows = img.rows;
    int cols = img.cols;

    // Perform convolution
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = 0.0;

            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int ni = i + ki;
                    int nj = j + kj;

                    // Check boundaries
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        sum += img.at<uint8_t>(ni, nj) * kernel[ki + halfSize][kj + halfSize];
                    }
                }
            }

            blurred.at<uint8_t>(i, j) = static_cast<uint8_t>(round(sum));
        }
    }

    return blurred;
}

// Function to add Additive White Gaussian Noise (AWGN)
Mat omp_apply_awgn(const Mat& img, double stddev) {
    /*Mat noise = Mat(img.size(), CV_64F);
    randn(noise, 0, stddev);
    Mat noisy_img;
    img.convertTo(noisy_img, CV_64F);
    noisy_img += noise;
    noisy_img = max(noisy_img, 0.0);
    noisy_img = min(noisy_img, 255.0);
    noisy_img.convertTo(noisy_img, CV_8U);
    return noisy_img;*/

    // Create a noise matrix of type double for precision
    Mat noise(img.size(), CV_64F);

    // Initialize Gaussian noise with mean=0 and standard deviation=stddev
    randn(noise, 0, stddev);

    // Convert the input image to double for precise addition
    Mat noisy_img;
    img.convertTo(noisy_img, CV_64F);

    int rows = noisy_img.rows;
    int cols = noisy_img.cols;

    // addition of noise
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Add noise to each pixel
            noisy_img.at<double>(i, j) += noise.at<double>(i, j);

            // Clamp the pixel values to the range [0, 255]
            noisy_img.at<double>(i, j) = std::min(std::max(noisy_img.at<double>(i, j), 0.0), 255.0);
        }
    }

    // Convert the noisy image back to 8-bit unsigned integer type
    noisy_img.convertTo(noisy_img, CV_8U);

    return noisy_img;
}

// Function to apply median filtering
Mat omp_apply_median_filter(const Mat& img, int kernel_size) {
    /*Mat filtered;
    medianBlur(img, filtered, kernel_size);
    return filtered;*/

    // Calculate the border size based on the kernel size
    int border = kernel_size / 2;

    // Pad the image to handle borders using reflection
    Mat padded;
    copyMakeBorder(img, padded, border, border, border, border, BORDER_REFLECT);

    // Initialize the filtered image as a clone of the original
    Mat filtered = img.clone();

    int rows = img.rows;
    int cols = img.cols;

    // Iterate over each pixel in the image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Vector to store the neighborhood pixel values
            vector<uchar> neighborhood;
            neighborhood.reserve(kernel_size * kernel_size);

            // Collect the neighborhood pixels
            for (int m = -border; m <= border; ++m) {
                for (int n = -border; n <= border; ++n) {
                    uchar pixel = padded.at<uchar>(i + border + m, j + border + n);
                    neighborhood.push_back(pixel);
                }
            }

            // Find the median using nth_element (does not fully sort the vector)
            size_t mid = neighborhood.size() / 2;
            nth_element(neighborhood.begin(), neighborhood.begin() + mid, neighborhood.end());
            uchar median = neighborhood[mid];

            // Assign the median value to the filtered image
            filtered.at<uchar>(i, j) = median;
        }
    }

    return filtered;
}

// Function to apply sharpening
Mat omp_apply_sharpen(const Mat& img, double sigma, double alpha) {
    /*Mat blurred, sharpened;
    GaussianBlur(img, blurred, Size(0, 0), sigma);
    sharpened = img + alpha * (img - blurred);
    sharpened = max(sharpened, 0.0);
    sharpened = min(sharpened, 255.0);
    sharpened.convertTo(sharpened, CV_8U);
    return sharpened;*/


    // Determine the kernel size based on sigma (common choice: 6*sigma +1)
    int kernel_size = static_cast<int>(std::ceil(6 * sigma)) | 1; // Ensure kernel_size is odd

    // Create Gaussian kernel
    std::vector<std::vector<double>> gaussian_kernel = omp_createGaussianKernel(kernel_size, sigma);
    int half = kernel_size / 2;

    // Pad the image to handle borders
    Mat padded;
    copyMakeBorder(img, padded, half, half, half, half, BORDER_REFLECT);

    // Initialize blurred image
    Mat blurred = Mat::zeros(img.size(), CV_64F);

    int rows = img.rows;
    int cols = img.cols;

    // Gaussian blur process 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double accum = 0.0;
            for (int m = -half; m <= half; ++m) {
                for (int n = -half; n <= half; ++n) {
                    uchar pixel = padded.at<uchar>(i + half + m, j + half + n);
                    accum += pixel * gaussian_kernel[m + half][n + half];
                }
            }
            blurred.at<double>(i, j) = accum;
        }
    }

    // Convert original image to double for precise calculations
    Mat img_double;
    img.convertTo(img_double, CV_64F);

    // Apply the sharpening formula: sharpened = original + alpha * (original - blurred)
    Mat sharpened = img_double + alpha * (img_double - blurred);

    // Clamp the pixel values to the range [0, 255] and convert back to 8-bit
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            sharpened.at<double>(i, j) = std::min(std::max(sharpened.at<double>(i, j), 0.0), 255.0);
        }
    }

    sharpened.convertTo(sharpened, CV_8U);

    return sharpened;
}

double omp_bilinearInterpolate(const Mat& img, double x, double y) {
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = std::min(x1 + 1, img.cols - 1);
    int y2 = std::min(y1 + 1, img.rows - 1);

    double dx = x - x1;
    double dy = y - y1;

    double val = (1 - dx) * (1 - dy) * img.at<uint8_t>(y1, x1) +
        dx * (1 - dy) * img.at<uint8_t>(y1, x2) +
        (1 - dx) * dy * img.at<uint8_t>(y2, x1) +
        dx * dy * img.at<uint8_t>(y2, x2);

    return val;
}

// Function to resize an image to a new size
Mat omp_resizeImage(const Mat& img, int newRows, int newCols) {
    Mat resized(newRows, newCols, img.type());
    double rowScale = static_cast<double>(img.rows) / newRows;
    double colScale = static_cast<double>(img.cols) / newCols;

    for (int r = 0; r < newRows; ++r) {
        for (int c = 0; c < newCols; ++c) {
            double origY = r * rowScale;
            double origX = c * colScale;

            resized.at<uint8_t>(r, c) = static_cast<uint8_t>(omp_bilinearInterpolate(img, origX, origY));
        }
    }

    return resized;
}

// Function to resize image (downscale and upscale)
Mat omp_apply_resize(const Mat& img, double scale) {
    int resizedRows = static_cast<int>(img.rows * scale);
    int resizedCols = static_cast<int>(img.cols * scale);

    // Resize the image using custom implementation
    Mat resized = omp_resizeImage(img, resizedRows, resizedCols);

    // Restore the image to the original size
    Mat restored = omp_resizeImage(resized, img.rows, img.cols);

    return restored;
}

// Function to perform Haar Wavelet Transform (DWT)
void omp_haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH) {
    int rows = src.rows / 2;
    int cols = src.cols / 2;

    LL = Mat(rows, cols, CV_64F, Scalar(0));
    LH = Mat(rows, cols, CV_64F, Scalar(0));
    HL = Mat(rows, cols, CV_64F, Scalar(0));
    HH = Mat(rows, cols, CV_64F, Scalar(0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double a = src.at<double>(i * 2, j * 2);
            double b = src.at<double>(i * 2, j * 2 + 1);
            double c = src.at<double>(i * 2 + 1, j * 2);
            double d = src.at<double>(i * 2 + 1, j * 2 + 1);

            LL.at<double>(i, j) = (a + b + c + d) / 4.0;
            LH.at<double>(i, j) = (a - b + c - d) / 4.0;
            HL.at<double>(i, j) = (a + b - c - d) / 4.0;
            HH.at<double>(i, j) = (a - b - c + d) / 4.0;
        }
    }
}

// Function to perform Inverse Haar Wavelet Transform (IDWT)
void omp_inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst) {
    int rows = LL.rows * 2;
    int cols = LL.cols * 2;

    dst = Mat(rows, cols, CV_64F, Scalar(0));

    for (int i = 0; i < LL.rows; ++i) {
        for (int j = 0; j < LL.cols; ++j) {
            double ll = LL.at<double>(i, j);
            double lh = LH.at<double>(i, j);
            double hl = HL.at<double>(i, j);
            double hh = HH.at<double>(i, j);

            dst.at<double>(i * 2, j * 2) = ll + lh + hl + hh;
            dst.at<double>(i * 2, j * 2 + 1) = ll - lh + hl - hh;
            dst.at<double>(i * 2 + 1, j * 2) = ll + lh - hl - hh;
            dst.at<double>(i * 2 + 1, j * 2 + 1) = ll - lh - hl + hh;
        }
    }
    // Normalize the values to [0,255]
    //normalize(dst, dst, 0, 255, NORM_MINMAX);
}

bool omp_is_image_size_sufficient(int original_width, int original_height, int watermark_width, int watermark_height, int block_size, int n_blocks_to_embed) {
    // Calculate the total required blocks for the watermark
    int total_required_blocks = ceil((double)(watermark_width * watermark_height) / (block_size * block_size));

    // Calculate the total available blocks in the original image
    int total_image_blocks = (original_width / block_size) * (original_height / block_size);

    // Use the minimum of available blocks and n_blocks_to_embed
    int available_blocks = min(total_image_blocks, n_blocks_to_embed);

    // Check if available blocks are sufficient
    return available_blocks >= total_required_blocks;
}

void omp_compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt) {
    // Convert OpenCV matrix to Eigen matrix
    Eigen::MatrixXd src_eigen;
    cv::cv2eigen(src, src_eigen);

    // Perform SVD using Eigen
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(src_eigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Assign results to OpenCV matrices
    cv::eigen2cv(svd.matrixU(), U);
    cv::eigen2cv(svd.singularValues(), S);
    cv::eigen2cv(svd.matrixV(), Vt);
}

void omp_reconstruct_matrix(const Mat& U, const Mat& S, const Mat& Vt, Mat& reconstructed) {
    // Convert OpenCV matrices to Eigen matrices
    Eigen::MatrixXd U_eigen, S_eigen, Vt_eigen;
    cv::cv2eigen(U, U_eigen);
    cv::cv2eigen(S, S_eigen);
    cv::cv2eigen(Vt, Vt_eigen);

    // Create a diagonal matrix from the singular values
    Eigen::MatrixXd S_diag = S_eigen;
    if (S.rows == 1 || S.cols == 1)
        S_diag = S_eigen.asDiagonal();

    // Reconstruct the matrix using Eigen's matrix multiplication
    Eigen::MatrixXd reconstructed_eigen = U_eigen * S_diag * Vt_eigen.transpose();
    // Convert the reconstructed Eigen matrix back to an OpenCV matrix
    cv::eigen2cv(reconstructed_eigen, reconstructed);

}

cv::Mat omp_customConvert8U(const cv::Mat& src) {
    Mat dst;

    // Step 1: Find the min and max values in the source matrix
    double minVal, maxVal;
    minMaxLoc(src, &minVal, &maxVal);

    // Step 2: Check if the values are within [0, 255]
    if (minVal < 0 || maxVal > 255) {
        // Step 3: Scale the values to fit within [0, 255]
        Mat normalized;
        src.convertTo(normalized, CV_64F, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

        // Convert to CV_8U
        normalized.convertTo(dst, CV_8U);
    }
    else {
        // Direct conversion if values are already within range
        src.convertTo(dst, CV_8U);
    }

    return dst;
}

cv::Mat omp_extractPrecisionDifference(const Mat& mat8u, const Mat& mat64f) {
    cv::Mat dst(mat64f.rows, mat64f.cols, mat64f.type());
#pragma omp parallel for
    for (int i = 0; i < mat64f.rows; ++i) {
        for (int j = 0; j < mat64f.cols; ++j) {
            double value = mat64f.at<double>(i, j);
            int integerPart = mat8u.at<uchar>(i, j);
            double fractionalPart = value - integerPart;
            dst.at<double>(i, j) = fractionalPart;
        }
    }

    return dst;
}

cv::Mat omp_combineMatPrecision(const cv::Mat& integerMat, const cv::Mat& precisionMat) {
    cv::Mat dst(precisionMat.rows, precisionMat.cols, CV_64F);
    for (int i = 0; i < precisionMat.rows; ++i) {
        for (int j = 0; j < precisionMat.cols; ++j) {
            dst.at<double>(i, j) = precisionMat.at<double>(i, j) + (integerMat.at<uchar>(i, j));
        }
    }

    return dst;
}

// Function to save the singular value matrix (S) as a secret key
void omp_save_singular_values(const Mat& S, const string& key_file) {
    ofstream file(key_file, ios::binary);
    if (file.is_open()) {
        int rows = S.rows;
        int cols = S.cols;
        file.write((char*)&rows, sizeof(int));
        file.write((char*)&cols, sizeof(int));
        file.write((char*)S.data, rows * cols * sizeof(double));
        file.close();
        cout << "Secret key (singular values) saved to " << key_file << endl;
    }
    else {
        cerr << "Error: Could not save singular values to file." << endl;
    }
}

// Function to load the singular value matrix (S) from the secret key file
Mat omp_load_singular_values(const string& key_file) {
    ifstream file(key_file, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not load singular values from file." << endl;
        return Mat();
    }

    int rows, cols;
    file.read((char*)&rows, sizeof(int));
    file.read((char*)&cols, sizeof(int));
    Mat S(rows, cols, CV_64F);
    file.read((char*)S.data, rows * cols * sizeof(double));
    file.close();
    return S;
}

void omp_save_selected_blocks(const vector<Block>& selected_blocks, const string& key_file) {
    std::ofstream file(key_file, std::ios::binary);

    if (file.is_open()) {
        // Write the number of selected blocks
        int num_blocks = selected_blocks.size();
        file.write((char*)&num_blocks, sizeof(int));

        // Write each block information
        for (const Block& block : selected_blocks) {
            // Write block location (x, y, width, height)
            file.write((char*)&block.location.x, sizeof(int));
            file.write((char*)&block.location.y, sizeof(int));
            file.write((char*)&block.location.width, sizeof(int));
            file.write((char*)&block.location.height, sizeof(int));

            // Write block's spatial and attack values
            file.write((char*)&block.spatial_value, sizeof(double));
            file.write((char*)&block.attack_value, sizeof(double));
        }

        file.close();
        std::cout << "Selected blocks information saved to " << key_file << std::endl;
    }
    else {
        std::cerr << "Error: Could not save selected blocks to file." << std::endl;
    }
}

vector<Block> omp_load_selected_blocks(const string& key_file) {
    vector<Block> selected_blocks;
    std::ifstream file(key_file, std::ios::binary);

    if (!file.is_open()) {
        // Handle error: couldn't open file
        throw std::runtime_error("Could not open key file");
    }

    // Read the number of selected blocks
    int num_blocks;
    file.read((char*)&num_blocks, sizeof(int));

    selected_blocks.resize(num_blocks);

    // Read each block information
    for (int i = 0; i < num_blocks; ++i) {
        Block block;
        file.read((char*)&block.location.x, sizeof(int));
        file.read((char*)&block.location.y, sizeof(int));
        file.read((char*)&block.location.width, sizeof(int));
        file.read((char*)&block.location.height, sizeof(int));
        file.read((char*)&block.spatial_value, sizeof(double));
        file.read((char*)&block.attack_value, sizeof(double));
        selected_blocks[i] = block;
    }

    file.close();
    return selected_blocks;
}

void omp_save_svd_components(const Mat& U, const Mat& S, const Mat& Vt, const string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "U" << U;
    fs << "S" << S;
    fs << "Vt" << Vt;

    fs.release();
}

void omp_load_svd_components(const string& filename, Mat& U, Mat& S, Mat& Vt) {
    FileStorage fs(filename, FileStorage::READ);

    fs["U"] >> U;
    fs["S"] >> S;
    fs["Vt"] >> Vt;

    fs.release();
}

void omp_saveMatVectorToFile(const std::vector<Mat>& mat_vector, const std::string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "MatVector" << "[";
    for (const auto& mat : mat_vector) {
        fs << mat;
    }
    fs << "]";

    fs.release();
}

std::vector<Mat> omp_loadMatVectorFromFile(const std::string& filename) {
    FileStorage fs(filename, FileStorage::READ);

    std::vector<Mat> mat_vector;
    FileNode node = fs["MatVector"];
    for (FileNodeIterator it = node.begin(); it != node.end(); ++it) {
        Mat mat;
        *it >> mat;
        mat_vector.push_back(mat);
    }

    fs.release();
    return mat_vector;
}

void omp_savePrecisionMat(const cv::Mat& precisionMat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write matrix dimensions
    file << precisionMat.rows << " " << precisionMat.cols << std::endl;

    // Write matrix elements with fixed precision
    for (int i = 0; i < precisionMat.rows; ++i) {
        for (int j = 0; j < precisionMat.cols; ++j) {
            file << std::fixed << std::setprecision(6) << precisionMat.at<double>(i, j) << " ";
        }
        file << std::endl;
    }

    file.close();
}

cv::Mat omp_loadPrecisionMat(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }

    int rows, cols;
    file >> rows >> cols;

    cv::Mat precisionMat(rows, cols, CV_64F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> precisionMat.at<double>(i, j);
        }
    }

    file.close();

    return precisionMat;
}

// Function to embed watermark
Mat omp_embed_watermark(
    const Mat& original, const Mat& watermark, double alpha,
    const string& key_filename, int wm_width, int wm_height,
    int n_blocks_to_embed, int block_size, double spatial_weight, std::chrono::milliseconds** execution_time
) {

    // Initialize variables
    Mat watermarked_image = original.clone();
    watermarked_image.convertTo(watermarked_image, CV_64F);

    // Initialize blank_image
    Mat blank_image = Mat::zeros(original.size(), CV_64F);

    // Apply various attacks and accumulate differences

    vector<double> blur_sigma_values = { 0.1, 0.5, 1, 2, 1.0, 2.0 };
    vector<int> median_kernel_sizes = { 3, 5, 7, 9, 11 };
    vector<double> awgn_std_values = { 0.1, 0.5, 2, 5, 10 };
    vector<double> sharpen_sigma_values = { 0.1, 0.5, 2, 10 };
    vector<double> sharpen_alpha_values = { 0.1, 0.5, 1, 2 };
    vector<double> resize_scale_values = { 0.5, 0.75, 0.9, 1.1, 1.5 };
    int num_threads = omp_get_max_threads();
    vector<Mat> local_diffs(num_threads, Mat::zeros(original.size(), CV_64F));

    auto embed_begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        Mat& local_diff = local_diffs[thread_id];

        // 1. Gaussian Blur
#pragma omp for schedule(static)
        for (int i = 0; i < blur_sigma_values.size(); i++) {
            Mat attacked = omp_apply_blur(original, blur_sigma_values[i]);
            Mat diff;
            absdiff(attacked, original, diff);
            diff.convertTo(diff, CV_64F);
            local_diff += diff;
        }

        // 2. Median Filtering
#pragma omp for schedule(static)
        for (int i = 0; i < median_kernel_sizes.size(); i++) {
            Mat attacked = omp_apply_median_filter(original, median_kernel_sizes[i]);
            Mat diff;
            absdiff(attacked, original, diff);
            diff.convertTo(diff, CV_64F);
            local_diff += diff;
        }

        // 3. Additive White Gaussian Noise
#pragma omp for schedule(static)
        for (int i = 0; i < awgn_std_values.size(); i++) {
            Mat attacked = omp_apply_awgn(original, awgn_std_values[i]);
            Mat diff;
            absdiff(attacked, original, diff);
            diff.convertTo(diff, CV_64F);
            local_diff += diff;
        }

        // 4. Sharpening
#pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < sharpen_sigma_values.size(); i++) {
            for (int j = 0; j < sharpen_alpha_values.size(); j++) {
                Mat attacked = omp_apply_sharpen(original, sharpen_sigma_values[i], sharpen_alpha_values[j]);
                Mat diff;
                absdiff(attacked, original, diff);
                diff.convertTo(diff, CV_64F);
                local_diff += diff;
            }
        }

        // 5. Resizing
#pragma omp for schedule(static)
        for (int i = 0; i < resize_scale_values.size(); i++) {
            Mat attacked = omp_apply_resize(original, resize_scale_values[i]);
            Mat diff;
            absdiff(attacked, original, diff);
            diff.convertTo(diff, CV_64F);
            local_diff += diff;
        }
        local_diffs[thread_id] = local_diff;
    }

    for (const auto& diff : local_diffs) {
        blank_image += diff;
    }

    // Block selection based on spatial and attack values
    vector<Block> blocks_to_watermark;
    std::vector<std::vector<Block>> thread_results(omp_get_max_threads());
#pragma omp parallel 
    {
        //private block for each threads
        int thread_id = omp_get_thread_num();
        std::vector<Block>& local_blocks_to_watermark = thread_results[thread_id];


#pragma omp for collapse(2)
        for (int i = 0; i < original.rows; i += block_size) {
            for (int j = 0; j < original.cols; j += block_size) {
                // Ensure the block is within image boundaries
                if ((i + block_size) > original.rows || (j + block_size) > original.cols)
                    continue;

                Mat block = original(Rect(j, i, block_size, block_size));
                Scalar mean_scalar = mean(block);
                double mean_val = mean_scalar[0];

                // Filter blocks based on mean intensity
                if (mean_val < 230 && mean_val > 10) {
                    // Compute spatial value (average)
                    double spatial_val = mean_val;

                    // Compute attack value from blank_image
                    Mat attack_block = blank_image(Rect(j, i, block_size, block_size));
                    Scalar attack_mean = mean(attack_block);
                    double attack_val = attack_mean[0];

                    // Create block structure
                    Block blk;
                    blk.location = Rect(j, i, block_size, block_size);
                    blk.spatial_value = spatial_val;
                    blk.attack_value = attack_val;
                    blk.merit = 0.0;
                    local_blocks_to_watermark.push_back(blk);
                    //blocks_to_watermark.push_back(blk);
                }
            }
        }
    }
    // Merge thread-local results
    for (const auto& local_blocks : thread_results) {
        blocks_to_watermark.insert(blocks_to_watermark.end(), local_blocks.begin(), local_blocks.end());
    }

    // Sort blocks based on spatial value (descending)
    sort(blocks_to_watermark.begin(), blocks_to_watermark.end(), [&](const Block& a, const Block& b) {
        return a.spatial_value > b.spatial_value;
        });

    // Assign merit based on spatial rank
#pragma omp parallel for
    for (int i = 0; i < blocks_to_watermark.size(); ++i) {
        blocks_to_watermark[i].merit += i * spatial_weight;
    }

    // Sort blocks based on attack value (ascending)
    sort(blocks_to_watermark.begin(), blocks_to_watermark.end(), [&](const Block& a, const Block& b) {
        return a.attack_value < b.attack_value;
        });

    // Assign merit based on attack rank
    double attack_weight = 1.0 - spatial_weight;
#pragma omp parallel for
    for (int i = 0; i < blocks_to_watermark.size(); ++i) {
        blocks_to_watermark[i].merit += i * attack_weight;
    }

    // Sort blocks based on total merit (descending)
    sort(blocks_to_watermark.begin(), blocks_to_watermark.end(), [&](const Block& a, const Block& b) {
        return a.merit > b.merit;
        });

    // Select top n_blocks_to_embed blocks
    vector<Block> selected_blocks(min(n_blocks_to_embed, (int)blocks_to_watermark.size()));

#pragma omp parallel for
    for (int i = 0; i < selected_blocks.size(); ++i) {
        selected_blocks[i] = blocks_to_watermark[i];
    }
    auto embed_end = std::chrono::high_resolution_clock::now();
    **execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(embed_end - embed_begin);


    // Precompute SVD of the watermark
    wm_width = min(watermark.cols, wm_width); // Set max watermark size dynamically
    wm_height = min(watermark.rows, wm_height);
    Mat watermark_resized;
    resize(watermark, watermark_resized, Size(wm_width, wm_height), 0, 0, INTER_LINEAR);


    //Save resized watermark image
    Mat save_watermark = watermark_resized.clone();
    save_watermark.convertTo(save_watermark, CV_8U);
    imwrite(key_filename + "_actualwatermark.tiff", save_watermark);

    watermark_resized.convertTo(watermark_resized, CV_64F, 1.0 / 255.0); // Normalize


    Mat Uwm, Swm, Vtwm;
    omp_compute_svd(watermark_resized, Uwm, Swm, Vtwm);
    omp_save_svd_components(Uwm, Swm, Vtwm, key_filename + "wm_svd");
    omp_save_selected_blocks(selected_blocks, key_filename + "_block");

    vector<Mat> original_Sc(selected_blocks.size());
    // Embed watermark into selected blocks
    for (int idx = 0; idx < selected_blocks.size(); ++idx) {
        Rect block_loc = selected_blocks[idx].location;
        Mat block = watermarked_image(block_loc).clone();
        // Perform DWT on the block
        Mat LL, LH, HL, HH;
        omp_haar_wavelet_transform(block, LL, LH, HL, HH);

        // Perform SVD on LL subband
        Mat Uc, Sc, Vtc;
        omp_compute_svd(LL, Uc, Sc, Vtc);

        // Embed watermark subset
        int shape_LL_tmp = Sc.rows; // Assuming rows represent shape_LL_tmp
        int start_idx = (idx * shape_LL_tmp) % Swm.rows; // Modulo ensures cyclic access to Swm
        int end_idx = start_idx + shape_LL_tmp;

        // Assign the Sc to the correct order index
        original_Sc.at(idx) = Sc.clone();

        // Create a subset of Swm with wrapping
        vector<double> Swm_subset(shape_LL_tmp);
        for (int i = 0; i < shape_LL_tmp; ++i) {
            Swm_subset[i] = Swm.at<double>((start_idx + i) % Swm.rows, 0); // Access column-wise
        }

        // Embed the watermark into Sc
        int min_size = std::min(Sc.rows, (int)Swm_subset.size());
        for (int i = 0; i < min_size; ++i) {
            Sc.at<double>(i) += alpha * Swm_subset[i];
        }

        // Perform inverse DWT
        Mat reconstructed_block, modified_LL;
        omp_reconstruct_matrix(Uc, Sc, Vtc, modified_LL);
        omp_inverse_haar_wavelet_transform(modified_LL, LH, HL, HH, reconstructed_block);

        // Replace the block in the watermarked image
        reconstructed_block.copyTo(watermarked_image(block_loc));
    }

    omp_saveMatVectorToFile(original_Sc, key_filename + "ori_s");
    Mat watermarked_image_int;
    watermarked_image.convertTo(watermarked_image_int, CV_8U);
    Mat precision_obj = omp_extractPrecisionDifference(watermarked_image_int, watermarked_image);
    omp_savePrecisionMat(precision_obj, key_filename + "_precision");

    return watermarked_image_int;
}

Mat omp_extract_watermark(const Mat& watermarked_int_image, const string& key_filename, int n_blocks_to_extract, int block_size, double alpha) {
    // Load singular values and selected blocks
    Mat Uwm, Swm, Vtwm;
    vector<Block> selected_blocks = omp_load_selected_blocks(key_filename + "_block");
    omp_load_svd_components(key_filename + "wm_svd", Uwm, Swm, Vtwm);
    vector<Mat> ori_S = omp_loadMatVectorFromFile(key_filename + "ori_s");
    Mat precision = omp_loadPrecisionMat(key_filename + "_precision");
    Mat watermarked_image = omp_combineMatPrecision(watermarked_int_image, precision);
    // Initialize watermark image
    Mat extracted_watermark_S = Mat::zeros(Swm.size(), CV_64F);

    int total_blocks = min(n_blocks_to_extract, (int)selected_blocks.size());
    // Iterate over the selected blocks to extract watermark components
    for (int idx = 0; idx < total_blocks; ++idx) {
        Rect block_loc = selected_blocks[idx].location;
        Mat block = watermarked_image(block_loc).clone();
        block.convertTo(block, CV_64F);

        // Perform DWT on the block
        Mat LL, LH, HL, HH;
        omp_haar_wavelet_transform(block, LL, LH, HL, HH);
        // Perform SVD on LL subband
        Mat Uc, Sc, Vtc;
        omp_compute_svd(LL, Uc, Sc, Vtc);

        // Extract singular values related to the watermark
        int min_size = std::min(Sc.rows, Swm.rows);
        Mat current_oriSc = ori_S[idx];

        for (int i = 0; i < min_size; ++i) {
            // Compute the original index in Swm
            int wm_index = (idx * Sc.rows + i) % Swm.rows;

            // Calculate extracted watermark singular value
            double extracted_value = (Sc.at<double>(i) - current_oriSc.at<double>(i)) / alpha;
            // Add the extracted value to the corresponding position in extracted_watermark_S
            extracted_watermark_S.at<double>(wm_index) += extracted_value;
        }
    }
    Mat extracted_watermark;
    omp_reconstruct_matrix(Uwm, extracted_watermark_S, Vtwm, extracted_watermark);
    // Normalize and convert to CV_8U
    normalize(extracted_watermark, extracted_watermark, 0, 255, NORM_MINMAX);
    extracted_watermark.convertTo(extracted_watermark, CV_8U);

    return extracted_watermark;
}

int omp(std::chrono::milliseconds* execution_time, double* psnr, bool isDisplay, string original_image_path, string watermark_image_path, int watermark_width, int watermark_height) {
    int original_width = 512;
    int original_height = 512;
    /*int watermark_width = 64;
    int watermark_height = 64;*/
    int block_size = 4;
    int n_blocks_to_embed = 128;
    double spatial_weight = 0.33;

    string output_filename = "omp_watermarked_image";
    string output_image_path = output_filename + ".tiff";

    if (!std::filesystem::exists(original_image_path)) {
        std::cerr << "File does not exist: " << original_image_path << std::endl;
        return -1;
    }

    // Load original image in grayscale
    Mat original_image = imread(original_image_path, IMREAD_GRAYSCALE);
    if (original_image.empty()) {
        cerr << "Error: Could not load original image from path: " << original_image_path << endl;
        return -1;
    }

    // Resize original image to 512x512 if not already
    if (original_image.rows < original_height || original_image.cols < original_width) {
        resize(original_image, original_image, Size(original_height, original_width), 0, 0, INTER_LINEAR);
        cout << "Original image resized to " << original_width << "x" << original_height << endl;
    }

    if (isDisplay) {
        // Display original image
        namedWindow("Resized Original Image", WINDOW_NORMAL);
        imshow("Resized Original Image", original_image);
        // Wait for a key press to proceed
        waitKey(0);
    }

    // Save original resized image
    bool isSaved = imwrite("resized_" + original_image_path, original_image);
    if (isSaved) {
        cout << "Original resized image saved as '" << output_image_path << "'." << endl;
    }
    else {
        cerr << "Error: Could not save Original resized image." << endl;
    }

    // Load watermark image in grayscale
    Mat watermark_image = imread(watermark_image_path, IMREAD_GRAYSCALE);
    if (watermark_image.empty()) {
        cerr << "Error: Could not load watermark image from path: " << watermark_image_path << endl;
        return -1;
    }

    if (isDisplay) {
        // Display watermark image
        namedWindow("Watermark Image", WINDOW_NORMAL);
        imshow("Watermark Image", watermark_image);
        // Wait for a key press to proceed
        waitKey(0);
    }

    double alpha = 5.11; // Embedding strength


    Mat watermarked_image = omp_embed_watermark(
        original_image, watermark_image, alpha, output_filename,
        watermark_width, watermark_height,
        n_blocks_to_embed, block_size, spatial_weight, &execution_time
    );



    if (isDisplay) {
        namedWindow("Watermarked Image", WINDOW_NORMAL);
        imshow("Watermarked Image", watermarked_image);
        waitKey(0);
    }

    // Save watermarked image
    isSaved = imwrite(output_image_path, watermarked_image, { cv::IMWRITE_TIFF_COMPRESSION, 1 });
    if (isSaved) {
        cout << "Watermarked image saved as '" << output_image_path << "'." << endl;
    }
    else {
        cerr << "Error: Could not save watermarked image." << endl;
    }

    //PSNR after Embedding
    Mat ori = imread("resized_" + original_image_path, IMREAD_UNCHANGED);
    Mat wm = imread(output_image_path, IMREAD_UNCHANGED);
    *psnr = PSNR(ori, wm);
    cout << "Sequential PSNR embedding : " << *psnr << endl;

    //Extraction
    if (!std::filesystem::exists(output_image_path)) {
        std::cerr << "File does not exist: " << output_image_path << std::endl;
        return -1;
    }

    Mat ext_watermarked_image = imread(output_image_path, IMREAD_UNCHANGED);
    if (ext_watermarked_image.empty()) {
        cerr << "Error: Could not load original image from path: " << output_image_path << endl;
        return -1;
    }

    //omp start extraction
    Mat extracted_watermark = omp_extract_watermark(ext_watermarked_image, output_filename, n_blocks_to_embed, block_size, alpha);

    if (isDisplay) {
        // Display the extracted watermark
        namedWindow("Extracted Watermark", WINDOW_NORMAL);
        imshow("Extracted Watermark", extracted_watermark);
        waitKey(0);
    }

    // Save the extracted watermark
    string extracted_watermark_path = "extracted_watermark.png";
    bool isWatermarkSaved = imwrite(extracted_watermark_path, extracted_watermark);
    if (isWatermarkSaved) {
        cout << "Extracted watermark saved as '" << extracted_watermark_path << "'." << endl;
    }
    else {
        cerr << "Error: Could not save the extracted watermark." << endl;
    }
    return 0;
}
