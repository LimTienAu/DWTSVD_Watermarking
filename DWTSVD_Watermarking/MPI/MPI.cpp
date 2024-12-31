#include "Mpi.h"
#include "Common_include.h"
#define PI (355.0 / 113.0)
// Function to serialize blocks (if necessary)
void serialize_blocks(const std::vector<Block>& blocks, std::vector<char>& buffer) {
    buffer.resize(blocks.size() * sizeof(Block));
    std::memcpy(buffer.data(), blocks.data(), blocks.size() * sizeof(Block));
}

// Function to deserialize blocks (if necessary)
void deserialize_blocks(const std::vector<char>& buffer, std::vector<Block>& blocks) {
    int num_blocks = buffer.size() / sizeof(Block);
    blocks.resize(num_blocks);
    std::memcpy(blocks.data(), buffer.data(), buffer.size());
}

vector<vector<double>> mpi_createGaussianKernel(int kernelSize, double sigma) {
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
Mat mpi_apply_blur(const Mat& img, double sigma, int rank, int size) {
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
    vector<vector<double>> kernel = mpi_createGaussianKernel(kernelSize, sigma);
    // Create output Mat
    Mat blurred = Mat::zeros(img.size(), img.type());

    int rows = img.rows;
    int cols = img.cols;

    // Perform convolution
    for (int i = 0; i < rows; ++i) {
        for (int j = rank; j < cols; j += size) {
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
Mat mpi_apply_awgn(const Mat& img, double stddev, int rank, int size) {
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
        for (int j = rank; j < cols; j += size) {
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
Mat mpi_apply_median_filter(const Mat& img, int kernel_size, int rank, int size) {
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
        for (int j = rank; j < cols; j += size) {
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
Mat mpi_apply_sharpen(const Mat& img, double sigma, double alpha, int rank, int size) {
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
    std::vector<std::vector<double>> gaussian_kernel = mpi_createGaussianKernel(kernel_size, sigma);
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
        for (int j = rank; j < cols; j += size) {
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

double mpi_bilinearInterpolate(const Mat& img, double x, double y) {
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
Mat mpi_resizeImage(const Mat& img, int newRows, int newCols, int rank, int size) {
    Mat resized(newRows, newCols, img.type());
    double rowScale = static_cast<double>(img.rows) / newRows;
    double colScale = static_cast<double>(img.cols) / newCols;

    for (int r = 0; r < newRows; ++r) {
        for (int c = rank; c < newCols; c += size) {
            double origY = r * rowScale;
            double origX = c * colScale;

            resized.at<uint8_t>(r, c) = static_cast<uint8_t>(mpi_bilinearInterpolate(img, origX, origY));
        }
    }

    return resized;
}

// Function to resize image (downscale and upscale)
Mat mpi_apply_resize(const Mat& img, double scale, int rank, int size) {
    int resizedRows = static_cast<int>(img.rows * scale);
    int resizedCols = static_cast<int>(img.cols * scale);


    // Resize the image using custom implementation
    Mat resized = mpi_resizeImage(img, resizedRows, resizedCols, rank, size);

    // Restore the image to the original size
    Mat restored = mpi_resizeImage(resized, img.rows, img.cols, rank, size);

    return restored;
}


//// Function to apply Gaussian blur
//Mat mpi_apply_blur(const Mat& img, double sigma) {
//    Mat blurred;
//    cv::GaussianBlur(img, blurred, Size(0, 0), sigma);
//    return blurred;
//}
//
//// Function to add Additive White Gaussian Noise (AWGN)
//Mat mpi_apply_awgn(const Mat& img, double stddev) {
//    Mat noise = Mat(img.size(), CV_64F);
//    randn(noise, 0, stddev);
//    Mat noisy_img;
//    img.convertTo(noisy_img, CV_64F);
//    noisy_img += noise;
//    noisy_img = max(noisy_img, 0.0);
//    noisy_img = min(noisy_img, 255.0);
//    noisy_img.convertTo(noisy_img, CV_8U);
//    return noisy_img;
//}
//
//// Function to apply median filtering
//Mat mpi_apply_median_filter(const Mat& img, int kernel_size) {
//    Mat filtered;
//    medianBlur(img, filtered, kernel_size);
//    return filtered;
//}
//
//// Function to apply sharpening
//Mat mpi_apply_sharpen(const Mat& img, double sigma, double alpha) {
//    Mat blurred, sharpened;
//    GaussianBlur(img, blurred, Size(0, 0), sigma);
//    sharpened = img + alpha * (img - blurred);
//    sharpened = max(sharpened, 0.0);
//    sharpened = min(sharpened, 255.0);
//    sharpened.convertTo(sharpened, CV_8U);
//    return sharpened;
//}
//
//// Function to resize image (downscale and upscale)
//Mat mpi_apply_resize(const Mat& img, double scale) {
//    Mat resized, restored;
//    resize(img, resized, Size(), scale, scale, INTER_LINEAR);
//    resize(resized, restored, img.size(), 0, 0, INTER_LINEAR);
//    return restored;
//}

// Function to perform Haar Wavelet Transform (DWT)
void mpi_haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH) {
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
void mpi_inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst) {
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

bool mpi_is_image_size_sufficient(int original_width, int original_height, int watermark_width, int watermark_height, int block_size, int n_blocks_to_embed) {
    // Calculate the total required blocks for the watermark
    int total_required_blocks = ceil((double)(watermark_width * watermark_height) / (block_size * block_size));

    // Calculate the total available blocks in the original image
    int total_image_blocks = (original_width / block_size) * (original_height / block_size);

    // Use the minimum of available blocks and n_blocks_to_embed
    int available_blocks = min(total_image_blocks, n_blocks_to_embed);

    // Check if available blocks are sufficient
    return available_blocks >= total_required_blocks;
}

void mpi_compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt) {
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

void mpi_reconstruct_matrix(const Mat& U, const Mat& S, const Mat& Vt, Mat& reconstructed) {
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

cv::Mat mpi_customConvert8U(const cv::Mat& src) {
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

cv::Mat mpi_extractPrecisionDifference(const Mat& mat8u, const Mat& mat64f) {
    cv::Mat dst(mat64f.rows, mat64f.cols, mat64f.type());

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

cv::Mat mpi_combineMatPrecision(const cv::Mat& integerMat, const cv::Mat& precisionMat) {
    cv::Mat dst(precisionMat.rows, precisionMat.cols, CV_64F);
    for (int i = 0; i < precisionMat.rows; ++i) {
        for (int j = 0; j < precisionMat.cols; ++j) {
            dst.at<double>(i, j) = precisionMat.at<double>(i, j) + (integerMat.at<uchar>(i, j));
        }
    }

    return dst;
}

// Function to save the singular value matrix (S) as a secret key
void mpi_save_singular_values(const Mat& S, const string& key_file) {
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
Mat mpi_load_singular_values(const string& key_file) {
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

void mpi_save_selected_blocks(const vector<Block>& selected_blocks, const string& key_file) {
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

vector<Block> mpi_load_selected_blocks(const string& key_file) {
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

void mpi_save_svd_components(const Mat& U, const Mat& S, const Mat& Vt, const string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "U" << U;
    fs << "S" << S;
    fs << "Vt" << Vt;

    fs.release();
}

void mpi_load_svd_components(const string& filename, Mat& U, Mat& S, Mat& Vt) {
    FileStorage fs(filename, FileStorage::READ);

    fs["U"] >> U;
    fs["S"] >> S;
    fs["Vt"] >> Vt;

    fs.release();
}

void mpi_saveMatVectorToFile(const std::vector<Mat>& mat_vector, const std::string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "MatVector" << "[";
    for (const auto& mat : mat_vector) {
        fs << mat;
    }
    fs << "]";

    fs.release();
}

std::vector<Mat> mpi_loadMatVectorFromFile(const std::string& filename) {
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

void mpi_savePrecisionMat(const cv::Mat& precisionMat, const std::string& filename) {
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

cv::Mat mpi_loadPrecisionMat(const std::string& filename) {
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



void mpi_broadcast_image(Mat& img, int root, MPI_Comm comm, int rank)
{
    // 1) Broadcast rows, cols, and type
    int rows = (rank == root) ? img.rows : 0;
    int cols = (rank == root) ? img.cols : 0;
    int type = (rank == root) ? img.type() : 0; // e.g., CV_8U

    MPI_Bcast(&rows, 1, MPI_INT, root, comm);
    MPI_Bcast(&cols, 1, MPI_INT, root, comm);
    MPI_Bcast(&type, 1, MPI_INT, root, comm);

    // If not root, create the image with the broadcasted dimensions/type
    if (rank != root) {
        img.create(rows, cols, type);
    }

    // 2) Broadcast the actual pixel data
    // For grayscale CV_8U, we have rows*cols pixels
    // If you have multi-channel images, you'll need to adapt accordingly.
    if (rows > 0 && cols > 0) {
        MPI_Bcast(img.ptr<uchar>(), rows * cols, MPI_UNSIGNED_CHAR, root, comm);
    }
}

void mpi_parallel_optimization(
    const Mat& original, Mat& blank_image,
    int block_size, double spatial_weight,
    int n_blocks_to_embed, vector<Block>& selected_blocks,
    int rank, int size)
{
    try {
        if (rank == 0) std::cout << "[Rank 0] Starting MPI processing...\n" << endl;

        // Attack parameters
        vector<double> blur_sigma_values = { 0.1, 0.5, 1, 2, 1.0, 2.0 };
        vector<int>    median_kernel_sizes = { 3, 5, 7, 9, 11 };
        vector<double> awgn_std_values = { 0.1, 0.5, 2, 5, 10 };
        vector<double> sharpen_sigma_values = { 0.1, 0.5, 2, 10 };
        vector<double> sharpen_alpha_values = { 0.1, 0.5, 1, 2 };
        vector<double> resize_scale_values = { 0.5, 0.75, 0.9, 1.1, 1.5 };

        Mat local_diff = Mat::zeros(original.size(), CV_64F);
        cout << "gaussian start" << endl;
        // ---------- Gaussian Blur ----------
        if (rank == 0) std::cout << "[Rank 0] Processing Gaussian Blur...\n";
        for (int i = 0; i < blur_sigma_values.size(); i++) {
            Mat attacked = mpi_apply_blur(original, blur_sigma_values[i], rank, size);

            // Use absdiff or squared difference
            local_diff += (attacked - original).mul(attacked - original);
        }

        // ---------- Median Filtering ----------
        if (rank == 0) std::cout << "[Rank 0] Processing Median Filtering...\n";
        for (int i = 0; i < median_kernel_sizes.size(); i++) {
            Mat attacked = mpi_apply_median_filter(original, median_kernel_sizes[i], rank, size);
            local_diff += (attacked - original).mul(attacked - original);
        }

        // ---------- AWGN ----------
        if (rank == 0) std::cout << "[Rank 0] Processing AWGN...\n";
        for (int i = 0; i < awgn_std_values.size(); i++) {
            Mat attacked = mpi_apply_awgn(original, awgn_std_values[i], rank, size);
            local_diff += (attacked - original).mul(attacked - original);
        }

        // ---------- Sharpening ----------
        if (rank == 0) std::cout << "[Rank 0] Processing Sharpening...\n" << endl;
        for (int i = 0; i < sharpen_sigma_values.size(); i++) {
            for (size_t j = 0; j < sharpen_alpha_values.size(); ++j) {
                Mat attacked = mpi_apply_sharpen(original, sharpen_sigma_values[i], sharpen_alpha_values[j], rank, size);
                local_diff += (attacked - original).mul(attacked - original);
            }
        }

        // ---------- Resizing ----------
        if (rank == 0) std::cout << "[Rank 0] Processing Resizing...\n" << endl;
        for (int i = 0; i < resize_scale_values.size(); i++) {
            Mat attacked = mpi_apply_resize(original, resize_scale_values[i], rank, size);
            local_diff += (attacked - original).mul(attacked - original);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ---------- Allreduce for Aggregation ----------
        if (!local_diff.isContinuous()) {
            local_diff = local_diff.clone();
        }

        // Possible chunking approach to reduce memory usage in large images
        const int chunk_size = 262144;
        double* data_ptr = local_diff.ptr<double>();
        int total_elements = static_cast<int>(local_diff.total());
        int num_chunks = (total_elements + chunk_size - 1) / chunk_size;

        if (rank == 0) std::cout << "[Rank 0] Reducing local_diff data...\n" << endl;
        for (int i = 0; i < num_chunks; ++i) {
            int current_chunk_size = std::min(chunk_size, total_elements - i * chunk_size);
            MPI_Allreduce(MPI_IN_PLACE, data_ptr + i * chunk_size,
                current_chunk_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        blank_image += local_diff;
        MPI_Barrier(MPI_COMM_WORLD);

        // ---------- Block Selection ----------
        if (rank == 0) std::cout << "[Rank 0] Performing block selection...\n";
        vector<Block> local_blocks;

        for (int i = rank * block_size; i < original.rows; i += size * block_size) {
            for (int j = 0; j < original.cols; j += block_size) {
                if ((i + block_size) > original.rows || (j + block_size) > original.cols)
                    continue;

                Mat block = original(Rect(j, i, block_size, block_size));
                Scalar mean_scalar = mean(block);
                double mean_val = mean_scalar[0];

                if (mean_val < 230 && mean_val > 10) {
                    Mat attack_block = blank_image(Rect(j, i, block_size, block_size));
                    Scalar attack_mean = mean(attack_block);
                    double attack_val = attack_mean[0];

                    Block blk;
                    blk.location = Rect(j, i, block_size, block_size);
                    blk.spatial_value = mean_val;
                    blk.attack_value = attack_val;
                    blk.merit = 0.0;
                    local_blocks.push_back(blk);
                }
            }
        }
        if (rank == 0) cout << "blokkkk2028" << endl;
        // ---------- Gather and Broadcast Blocks ----------
        vector<Block> all_blocks, blocks0;

        // For simplicity, gather everything on rank 0
        if (rank == 0) {
            cout << "Moment be4 disaster " << endl;
            blocks0.insert(blocks0.end(), local_blocks.begin(), local_blocks.end());
        }
        if (rank == 0) cout << "2036" << endl;
        // *** If you truly want to gather blocks from all ranks, use MPI_Gatherv. 
        // Below is a simple Bcast approach if only rank 0 has blocks. ***

        // Barrier ensures local_blocks is fully formed
        MPI_Barrier(MPI_COMM_WORLD);

        // Serialize local_blocks to a byte buffer
        std::vector<char> send_buffer;
        serialize_blocks(local_blocks, send_buffer);
        int send_count = send_buffer.size();

        // Gather send counts from all ranks to rank 0
        std::vector<int> recv_counts(size, 0);
        MPI_Gather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Compute displacements and total receive size on rank 0
        std::vector<int> displs(size, 0);
        int total_recv = 0;
        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                displs[i] = total_recv;
                total_recv += recv_counts[i];
            }
        }

        // Receive buffer on rank 0
        std::vector<char> recv_buffer;
        if (rank == 0) {
            recv_buffer.resize(total_recv);
        }

        // Gather all serialized blocks to rank 0
        MPI_Gatherv(send_buffer.data(), send_count, MPI_CHAR,
            recv_buffer.data(), recv_counts.data(), displs.data(), MPI_CHAR,
            0, MPI_COMM_WORLD);

        // Deserialize blocks on rank 0
        if (rank == 0) {
            cout << "Moment be4 disaster2 " << endl;
            deserialize_blocks(recv_buffer, all_blocks);
            all_blocks.insert(all_blocks.end(), blocks0.begin(), blocks0.end());
        }



        if (rank == 0) cout << "2046" << endl;
        // If rank 0, compute merits
        if (rank == 0) {
            for (size_t i = 0; i < all_blocks.size(); ++i) {
                // Simple example: add partial merit based on index
                all_blocks[i].merit += i * spatial_weight;
            }

            // Sort descending by 'merit'
            std::sort(all_blocks.begin(), all_blocks.end(),
                [](const Block& a, const Block& b) {
                    return a.merit > b.merit;
                });

            // Select top n_blocks_to_embed
            if (static_cast<int>(all_blocks.size()) < n_blocks_to_embed) {
                // If fewer blocks than n_blocks_to_embed, just take them all
                n_blocks_to_embed = (int)all_blocks.size();
            }
            selected_blocks.assign(all_blocks.begin(), all_blocks.begin() + n_blocks_to_embed);
        }
        if (rank == 0) cout << "2067" << endl;
        if (rank == 0) std::cout << "[Rank 0] Block selection completed.\n" << endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " encountered an error: "
            << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

Mat mpi_embed_watermark(
    std::chrono::milliseconds** time,
    const Mat& original,
    const Mat& watermark,
    double alpha,
    const string& key_filename,
    int wm_width,
    int wm_height,
    int n_blocks_to_embed,
    int block_size,
    double spatial_weight,
    int rank,
    int size
) {
    Mat watermarked_image = original.clone();
    watermarked_image.convertTo(watermarked_image, CV_64F);

    // Initialize blank_image
    Mat blank_image = Mat::zeros(original.size(), CV_64F);

    // Gather blocks & compute blank_image differences
    vector<Block> selected_blocks;
    auto embed_begin = std::chrono::high_resolution_clock::now();
    mpi_parallel_optimization(
        original, blank_image,
        block_size, spatial_weight,
        n_blocks_to_embed, selected_blocks,
        rank, size
    );
    auto embed_end = std::chrono::high_resolution_clock::now();
    **time = std::chrono::duration_cast<std::chrono::milliseconds>(embed_end - embed_begin);
    // Resize/normalize the watermark on rank 0
    Mat watermark_resized;
    int wrows = 0, wcols = 0; // to broadcast dims to all ranks
    if (rank == 0) {
        wm_width = std::min(watermark.cols, wm_width);
        wm_height = std::min(watermark.rows, wm_height);

        resize(watermark, watermark_resized, Size(wm_width, wm_height), 0, 0, INTER_LINEAR);
        watermark_resized.convertTo(watermark_resized, CV_64F, 1.0 / 255.0);

        // Save the resized watermark to disk
        Mat save_watermark = watermark_resized.clone();
        normalize(save_watermark, save_watermark, 0, 255, NORM_MINMAX);
        save_watermark.convertTo(save_watermark, CV_8U);
        imwrite(key_filename + "_actualwatermark.tiff", save_watermark);

        wrows = watermark_resized.rows;
        wcols = watermark_resized.cols;

        cout << "done parallel" << rank << " " << size << endl;

        watermark_resized.create(wrows, wcols, CV_64F);


        // On rank 0, compute SVD and save
        Mat Uwm, Swm, Vtwm;
        mpi_compute_svd(watermark_resized, Uwm, Swm, Vtwm);
        mpi_save_svd_components(Uwm.clone(), Swm.clone(), Vtwm.clone(), key_filename + "wm_svd");
        mpi_save_selected_blocks(selected_blocks, key_filename + "_block");


        vector<Mat> original_Sc;
        for (size_t idx = 0; idx < selected_blocks.size(); ++idx) {
            Rect block_loc = selected_blocks[idx].location;
            Mat block = watermarked_image(block_loc).clone();

            Mat LL, LH, HL, HH;
            mpi_haar_wavelet_transform(block, LL, LH, HL, HH);

            Mat Uc, Sc, Vtc;
            mpi_compute_svd(LL, Uc, Sc, Vtc);

            original_Sc.push_back(Sc.clone());

            int shape_LL_tmp = Sc.rows;
            // We assume here that Swm is a column vector.
            // We must broadcast Swm or ensure it's used only on rank 0. 
            // For now, we assume all ranks have it from the prior broadcast. (We didn't broadcast it, but let's assume.)
            // More robust approach: Broadcast S, U, V if needed.
            // We skip that detail for brevity.

            int start_idx = (idx * shape_LL_tmp) % Swm.rows;

            vector<double> Swm_subset(shape_LL_tmp);
            for (int i = 0; i < shape_LL_tmp; ++i) {
                // Modded index to wrap around S
                Swm_subset[i] = Swm.at<double>((start_idx + i) % Swm.rows, 0);
            }

            int min_size = std::min(Sc.rows, static_cast<int>(Swm_subset.size()));
            for (int i = 0; i < min_size; ++i) {
                Sc.at<double>(i) += alpha * Swm_subset[i];
            }

            Mat modified_LL, reconstructed_block;
            mpi_reconstruct_matrix(Uc, Sc, Vtc, modified_LL);
            mpi_inverse_haar_wavelet_transform(modified_LL, LH, HL, HH, reconstructed_block);

            reconstructed_block.copyTo(watermarked_image(block_loc));
        }

        mpi_saveMatVectorToFile(original_Sc, key_filename + "ori_s");

        Mat watermarked_image_int;
        watermarked_image.convertTo(watermarked_image_int, CV_8U);
        Mat precision_obj = mpi_extractPrecisionDifference(watermarked_image_int, watermarked_image);
        mpi_savePrecisionMat(precision_obj, key_filename + "_precision");


        return watermarked_image;
    }
}

int mpi(std::chrono::milliseconds* time, double* psnr,
    bool isDisplay,
    string original_image_path,
    string watermark_image_path,
    int rank,
    int size)
{
    int original_width = 512;
    int original_height = 512;
    int watermark_width = 64;
    int watermark_height = 64;
    int block_size = 4;
    int n_blocks_to_embed = 128;
    double spatial_weight = 0.33;
    double alpha = 5.11;

    string output_filename = "mpi_watermarked_image";
    string output_image_path = output_filename + ".tiff";

    // ----------------------------------------
    // 1) Only rank 0 reads the images from disk
    // ----------------------------------------
    Mat original_image, watermark_image;
    if (rank == 0) {
        if (!std::filesystem::exists(original_image_path)) {
            std::cerr << "[Rank 0] File does not exist: " << original_image_path << std::endl;
            return -1;
        }
        original_image = imread(original_image_path, IMREAD_GRAYSCALE);
        if (original_image.empty()) {
            cerr << "[Rank 0] Error: Could not load original image: " << original_image_path << endl;
            return -1;
        }

        // Resize to 512�512
        resize(original_image, original_image, Size(original_width, original_height), 0, 0, INTER_LINEAR);

        if (!std::filesystem::exists(watermark_image_path)) {
            std::cerr << "[Rank 0] File does not exist: " << watermark_image_path << std::endl;
            return -1;
        }

        bool isSaved = imwrite("resized_" + original_image_path, original_image);
        if (isSaved) {
            cout << "Original resized image saved as '" << output_image_path << "'." << endl;
        }
        else {
            cerr << "Error: Could not save Original resized image." << endl;
        }

        watermark_image = imread(watermark_image_path, IMREAD_GRAYSCALE);
        if (watermark_image.empty()) {
            cerr << "[Rank 0] Error: Could not load watermark image: " << watermark_image_path << endl;
            return -1;
        }

        // Optionally display
        if (isDisplay) {
            imshow("Original Image", original_image);
            waitKey(0);
            imshow("Watermark Image", watermark_image);
            waitKey(0);
        }
    }

    // ----------------------------------------
    // 2) Broadcast both images so all ranks have them
    // ----------------------------------------
    mpi_broadcast_image(original_image, 0, MPI_COMM_WORLD, rank);
    mpi_broadcast_image(watermark_image, 0, MPI_COMM_WORLD, rank);

    // Ensure all ranks see the same data
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "22222" << endl;
    // ----------------------------------------
    // Embedding
    // ----------------------------------------

    Mat watermarked_image = mpi_embed_watermark(
        &time,
        original_image,
        watermark_image,
        alpha,
        output_filename,
        watermark_width,
        watermark_height,
        n_blocks_to_embed,
        block_size,
        spatial_weight,
        rank,
        size
    );

    if (rank == 0) {


        // Save final watermarked image
        imwrite(output_image_path, watermarked_image);
        std::cout << "[Rank 0] Watermarked image saved as '" << output_image_path << "'.\n" << endl;

        //PSNR after Embedding
        Mat ori = imread("resized_" + original_image_path, IMREAD_UNCHANGED);
        Mat wm = imread(output_image_path, IMREAD_UNCHANGED);

        ori.convertTo(ori, CV_64F);
        wm.convertTo(wm, CV_64F);
        cout << "ori: " << ori.size() << "  wm : " << wm.size() << endl;
        *psnr = PSNR(ori, wm);
        cout << "MPI PSNR embedding : " << *psnr << endl;


        //if (!std::filesystem::exists(output_image_path)) {
        //    std::cerr << "File does not exist: " << output_image_path << std::endl;
        //    return -1;
        //}

        //Mat ext_watermarked_image = imread(output_image_path, IMREAD_UNCHANGED);
        //if (ext_watermarked_image.empty()) {
        //    cerr << "Error: Could not load original image from path: " << output_image_path << endl;
        //    return -1;
        //}

        //Mat extracted_watermark = extract_watermark(ext_watermarked_image, output_filename, n_blocks_to_embed, block_size, alpha);

        //if (isDisplay) {
        //    // Display the extracted watermark
        //    namedWindow("Extracted Watermark", WINDOW_NORMAL);
        //    imshow("Extracted Watermark", extracted_watermark);
        //    waitKey(0);
        //}

        //// Save the extracted watermark
        //string extracted_watermark_path = "mpi_extracted_watermark.png";
        //bool isWatermarkSaved = imwrite(extracted_watermark_path, extracted_watermark);
        //if (isWatermarkSaved) {
        //    //cout << "Extracted watermark saved as '" << extracted_watermark_path << "'." << endl;
        //}
        //else {
        //    cerr << "Error: Could not save the extracted watermark." << endl;
        //}
    }

    return 0;
}




