#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <chrono>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <random>

// Structure to hold block information
struct Block {
    int x;
    int y;
    double spatial_value;
    double attack_value;
    double merit;
};

class WatermarkEmbedder {
public:
    // Function to compute wPSNR
    double wpsnr(const cv::Mat& img1, const cv::Mat& img2, const std::string& csf_path) {
        // Convert images to float and normalize
        std::vector<std::vector<float>> img1_f = matToFloatVector(img1);
        std::vector<std::vector<float>> img2_f = matToFloatVector(img2);

        // Compute difference
        std::vector<std::vector<float>> difference = subtractImages(img1_f, img2_f);

        // Check if images are identical
        bool same = true;
        for (size_t i = 0; i < difference.size() && same; ++i) {
            for (size_t j = 0; j < difference[0].size() && same; ++j) {
                if (difference[i][j] != 0.0f) {
                    same = false;
                }
            }
        }
        if (same) {
            return 9999999.0;
        }

        // Load CSF from CSV
        std::vector<std::vector<double>> csf = loadCSV(csf_path);
        if (csf.empty()) {
            std::cerr << "Error: CSF data is empty!" << std::endl;
            return -1.0;
        }

        // Perform convolution (valid mode)
        std::vector<std::vector<double>> ew = convolve2D(difference, csf);

        // Compute Mean Squared Error
        double mse = 0.0;
        size_t count = 0;
        for (const auto& row : ew) {
            for (const auto& val : row) {
                mse += val * val;
                count++;
            }
        }
        mse /= static_cast<double>(count);

        // Compute PSNR in decibels
        double decibels = 20.0 * log10(1.0 / std::sqrt(mse));
        return decibels;
    }

    // Function to apply Gaussian blur manually
    std::vector<std::vector<unsigned char>> blur(const cv::Mat& img, double sigma) {
        int kernel_size = std::ceil(sigma * 6);
        if (kernel_size % 2 == 0) kernel_size += 1; // Ensure it's odd
        std::vector<std::vector<double>> kernel = generateGaussianKernel(kernel_size, sigma);
        return convolveImage(img, kernel);
    }

    // Function to add AWGN manually
    std::vector<std::vector<unsigned char>> awgn(const cv::Mat& img, double std_dev, unsigned int seed) {
        std::vector<std::vector<unsigned char>> noise_img = imageToVector(img);
        std::mt19937 generator(seed);
        std::normal_distribution<double> distribution(0.0, std_dev);

        for (auto& row : noise_img) {
            for (auto& pixel : row) {
                double noise = distribution(generator);
                int new_val = static_cast<int>(pixel + noise);
                new_val = std::min(std::max(new_val, 0), 255);
                pixel = static_cast<unsigned char>(new_val);
            }
        }
        return noise_img;
    }

    // Function to apply sharpening manually
    cv::Mat sharpening(const cv::Mat& img, double sigma, double alpha) {
        // Generate Gaussian kernel
        int kernel_size = std::ceil(sigma * 6);
        if (kernel_size % 2 == 0) kernel_size += 1; // Ensure it's odd
        std::vector<std::vector<double>> kernel = generateGaussianKernel(kernel_size, sigma);

        // Blur the image
        std::vector<std::vector<unsigned char>> blurred = convolveImage(img, kernel);

        // Create sharpened image
        cv::Mat attacked(img.size(), CV_8UC1);
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                double val = static_cast<double>(img.at<uchar>(i, j)) +
                    alpha * (static_cast<double>(img.at<uchar>(i, j)) - static_cast<double>(blurred[i][j]));
                val = std::min(std::max(val, 0.0), 255.0);
                attacked.at<uchar>(i, j) = static_cast<unsigned char>(val);
            }
        }
        return attacked;
    }

    // Function to apply median filter manually
    std::vector<std::vector<unsigned char>> median_filter(const cv::Mat& img, int kernel_size) {
        return applyMedianFilter(img, kernel_size);
    }

    // Function to apply resizing manually using bilinear interpolation
    cv::Mat resizing(const cv::Mat& img, double scale) {
        return resizeImage(img, scale);
    }

    // Function to perform Haar DWT (1-level)
    std::vector<Eigen::MatrixXd> dwt(const Eigen::MatrixXd& img) {
        int rows = img.rows();
        int cols = img.cols();
        int rows_ll = rows / 2;
        int cols_ll = cols / 2;

        Eigen::MatrixXd LL = Eigen::MatrixXd::Zero(rows_ll, cols_ll);
        Eigen::MatrixXd LH = Eigen::MatrixXd::Zero(rows_ll, cols_ll);
        Eigen::MatrixXd HL = Eigen::MatrixXd::Zero(rows_ll, cols_ll);
        Eigen::MatrixXd HH = Eigen::MatrixXd::Zero(rows_ll, cols_ll);

        for (int i = 0; i < rows_ll; ++i) {
            for (int j = 0; j < cols_ll; ++j) {
                double a = img(2 * i, 2 * j);
                double b = img(2 * i, 2 * j + 1);
                double c = img(2 * i + 1, 2 * j);
                double d = img(2 * i + 1, 2 * j + 1);

                LL(i, j) = (a + b + c + d) / 2.0;
                LH(i, j) = (a - b + c - d) / 2.0;
                HL(i, j) = (a + b - c - d) / 2.0;
                HH(i, j) = (a - b - c + d) / 2.0;
            }
        }

        return { LL, LH, HL, HH };
    }

    // Function to perform inverse Haar DWT (1-level)
    Eigen::MatrixXd idwt(const std::vector<Eigen::MatrixXd>& coeffs) {
        const Eigen::MatrixXd& LL = coeffs[0];
        const Eigen::MatrixXd& LH = coeffs[1];
        const Eigen::MatrixXd& HL = coeffs[2];
        const Eigen::MatrixXd& HH = coeffs[3];

        int rows_ll = LL.rows();
        int cols_ll = LL.cols();
        int rows = rows_ll * 2;
        int cols = cols_ll * 2;

        Eigen::MatrixXd img = Eigen::MatrixXd::Zero(rows, cols);

        for (int i = 0; i < rows_ll; ++i) {
            for (int j = 0; j < cols_ll; ++j) {
                double a = LL(i, j) + LH(i, j) + HL(i, j) + HH(i, j);
                double b = LL(i, j) - LH(i, j) + HL(i, j) - HH(i, j);
                double c = LL(i, j) + LH(i, j) - HL(i, j) - HH(i, j);
                double d = LL(i, j) - LH(i, j) - HL(i, j) + HH(i, j);

                img(2 * i, 2 * j) = a / 2.0;
                img(2 * i, 2 * j + 1) = b / 2.0;
                img(2 * i + 1, 2 * j) = c / 2.0;
                img(2 * i + 1, 2 * j + 1) = d / 2.0;
            }
        }

        return img;
    }

    // Function to perform SVD using Eigen
    void compute_svd(const Eigen::MatrixXd& mat, Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& Vt) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        S = svd.singularValues();
        Vt = svd.matrixV().transpose();
    }

    // Function to embed watermark
    cv::Mat embedding(const std::string& original_image_path, const std::string& watermark_image_path) {
        // Load original image in grayscale
        cv::Mat original_image = cv::imread(original_image_path, cv::IMREAD_GRAYSCALE);
        if (original_image.empty()) {
            std::cerr << "Error: Cannot load original image!" << std::endl;
            return original_image;
        }

        // Display original image
        cv::imshow("Original Image", original_image);
        cv::waitKey(0);

        // Load watermark image in grayscale
        cv::Mat watermark_image = cv::imread(watermark_image_path, cv::IMREAD_GRAYSCALE);
        if (watermark_image.empty()) {
            std::cerr << "Error: Cannot load watermark image!" << std::endl;
            return original_image;
        }

        // Resize watermark image to fit embedding blocks if necessary
        // For flexibility, we will map the entire watermark image into the embedding blocks
        int watermark_rows = watermark_image.rows;
        int watermark_cols = watermark_image.cols;

        // Convert watermark image to vector<double>
        std::vector<double> watermark_to_embed;
        for (int i = 0; i < watermark_image.rows; ++i) {
            for (int j = 0; j < watermark_image.cols; ++j) {
                // Normalize watermark pixel values to range [0,1]
                double pixel_val = static_cast<double>(watermark_image.at<uchar>(i, j)) / 255.0;
                watermark_to_embed.push_back(pixel_val);
            }
        }

        double alpha = 5.11;
        int n_blocks_to_embed = watermark_to_embed.size(); // Number of blocks equals number of watermark pixels
        int block_size = 4;
        std::string spatial_function = "average";
        double spatial_weight = 0.33;
        double attack_weight = 1.0 - spatial_weight;

        std::vector<Block> blocks_to_watermark;
        std::vector<std::vector<double>> blank_image(original_image.rows, std::vector<double>(original_image.cols, 0.0));

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Apply various attacks and accumulate differences
        // Gaussian Blur
        std::vector<double> blur_sigma_values = { 0.1, 0.5, 1, 2 };
        for (double sigma : blur_sigma_values) {
            std::vector<std::vector<unsigned char>> attacked_vec = blur(original_image, sigma);
            accumulateDifferences(attacked_vec, original_image, blank_image);
        }

        // Median Filtering
        std::vector<int> kernel_sizes = { 3, 5, 7, 9, 11 };
        for (int k : kernel_sizes) {
            std::vector<std::vector<unsigned char>> attacked_vec = median_filter(original_image, k);
            accumulateDifferences(attacked_vec, original_image, blank_image);
        }

        // Additive White Gaussian Noise
        std::vector<double> awgn_std = { 0.1, 0.5, 2, 5, 10 };
        for (double std_dev : awgn_std) {
            std::vector<std::vector<unsigned char>> attacked_vec = awgn(original_image, std_dev, 0);
            accumulateDifferences(attacked_vec, original_image, blank_image);
        }

        // Sharpening
        std::vector<double> sharpening_sigma_values = { 0.1, 0.5, 2, 100 };
        std::vector<double> sharpening_alpha_values = { 0.1, 0.5, 1, 2 };
        for (double sigma : sharpening_sigma_values) {
            for (double alpha_sharp : sharpening_alpha_values) {
                cv::Mat attacked = sharpening(original_image, sigma, alpha_sharp);
                std::vector<std::vector<unsigned char>> attacked_vec = imageToVector(attacked);
                accumulateDifferences(attacked_vec, original_image, blank_image);
            }
        }

        // Resizing
        std::vector<double> resizing_scale_values = { 0.5, 0.75, 0.9, 1.1, 1.5 };
        for (double scale : resizing_scale_values) {
            cv::Mat attacked = resizing(original_image, scale);
            std::vector<std::vector<unsigned char>> attacked_vec = imageToVector(attacked);
            accumulateDifferences(attacked_vec, original_image, blank_image);
        }

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "[EMBEDDING] Time of attacks for embedding: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "[EMBEDDING] Spatial function: " << spatial_function << std::endl;

        // Find blocks to watermark based on spatial and attack values
        for (int i = 0; i < original_image.rows; i += block_size) {
            for (int j = 0; j < original_image.cols; j += block_size) {
                if (i + block_size > original_image.rows || j + block_size > original_image.cols)
                    continue;

                // Extract block
                std::vector<std::vector<unsigned char>> block = getBlock(original_image, i, j, block_size);
                double mean_val = computeMean(block);
                if (mean_val < 230 && mean_val > 10) {
                    double spatial_val = 0.0;
                    if (spatial_function == "average") {
                        spatial_val = computeMean(block);
                    }
                    // Additional spatial functions can be added here

                    // Compute attack value
                    double attack_val = computeAttackValue(blank_image, i, j, block_size);

                    Block blk;
                    blk.x = i;
                    blk.y = j;
                    blk.spatial_value = spatial_val;
                    blk.attack_value = attack_val;
                    blk.merit = 0.0;
                    blocks_to_watermark.push_back(blk);
                }
            }
        }

        // Sort blocks based on spatial value (descending)
        std::sort(blocks_to_watermark.begin(), blocks_to_watermark.end(),
            [&](const Block& a, const Block& b) -> bool {
                return a.spatial_value > b.spatial_value;
            });

        // Assign merit based on spatial weight
        for (size_t i = 0; i < blocks_to_watermark.size(); ++i) {
            blocks_to_watermark[i].merit += i * spatial_weight;
        }

        // Sort blocks based on attack value (ascending)
        std::sort(blocks_to_watermark.begin(), blocks_to_watermark.end(),
            [&](const Block& a, const Block& b) -> bool {
                return a.attack_value < b.attack_value;
            });

        // Assign merit based on attack weight
        for (size_t i = 0; i < blocks_to_watermark.size(); ++i) {
            blocks_to_watermark[i].merit += i * attack_weight;
        }

        // Sort blocks based on merit (descending)
        std::sort(blocks_to_watermark.begin(), blocks_to_watermark.end(),
            [&](const Block& a, const Block& b) -> bool {
                return a.merit > b.merit;
            });

        // Select top n_blocks_to_embed
        std::vector<Block> blocks_final;
        for (int i = 0; i < n_blocks_to_embed && i < blocks_to_watermark.size(); ++i) {
            blocks_final.push_back(blocks_to_watermark[i]);
            // Update blank_image to mark embedded blocks
            for (int m = blocks_to_watermark[i].x; m < blocks_to_watermark[i].x + block_size; ++m) {
                for (int n = blocks_to_watermark[i].y; n < blocks_to_watermark[i].y + block_size; ++n) {
                    blank_image[m][n] = 1.0;
                }
            }
        }

        // Sort final blocks based on location (ascending)
        std::sort(blocks_final.begin(), blocks_final.end(),
            [&](const Block& a, const Block& b) -> bool {
                if (a.x != b.x)
                    return a.x < b.x;
                return a.y < b.y;
            });

        double divisions = static_cast<double>(original_image.rows) / block_size;
        int shape_LL_tmp = static_cast<int>(std::floor(original_image.rows / (2 * divisions)));

        cv::Mat watermarked_image = original_image.clone();

        // Reshape watermark
        // Create a matrix from watermark data
        int wm_rows = watermark_image.rows;
        int wm_cols = watermark_image.cols;
        Eigen::MatrixXd watermark_mat(wm_rows, wm_cols);
        for (int i = 0; i < wm_rows; ++i) {
            for (int j = 0; j < wm_cols; ++j) {
                watermark_mat(i, j) = watermark_to_embed[i * wm_cols + j];
            }
        }

        Eigen::MatrixXd Uwm, Vwm;
        Eigen::VectorXd Swm;
        compute_svd(watermark_mat, Uwm, Swm, Vwm);

        for (size_t i = 0; i < blocks_final.size(); ++i) {
            int x = blocks_final[i].x;
            int y = blocks_final[i].y;

            // Extract block as Eigen matrix
            Eigen::MatrixXd block_eigen = getBlockEigen(original_image, x, y, block_size);

            // Perform DWT
            std::vector<Eigen::MatrixXd> coeffs = dwt(block_eigen);
            Eigen::MatrixXd LL = coeffs[0];

            // Perform SVD on LL
            Eigen::MatrixXd Uc, Vc;
            Eigen::VectorXd Sc;
            compute_svd(LL, Uc, Sc, Vc);

            // Embed watermark
            // Ensure we don't exceed the watermark vector
            if ((i * shape_LL_tmp) < Swm.size()) {
                for (int m = 0; m < Sc.size() && (i * shape_LL_tmp + m) < Swm.size(); ++m) {
                    Sc(m) += Swm(m) * alpha;
                }
            }

            // Reconstruct LL
            Eigen::MatrixXd LL_new = Uc * Sc.asDiagonal() * Vc;

            // Perform inverse DWT
            coeffs[0] = LL_new;
            Eigen::MatrixXd block_new_eigen = idwt(coeffs);

            // Clip values to [0, 255]
            for (int m = 0; m < block_size; ++m) {
                for (int n = 0; n < block_size; ++n) {
                    block_new_eigen(m, n) = std::min(std::max(block_new_eigen(m, n), 0.0), 255.0);
                }
            }

            // Convert back to cv::Mat and replace in watermarked_image
            cv::Mat block_new(block_size, block_size, CV_64F);
            for (int m = 0; m < block_size; ++m) {
                for (int n = 0; n < block_size; ++n) {
                    block_new.at<double>(m, n) = block_new_eigen(m, n);
                }
            }
            block_new.convertTo(block_new, CV_8U);
            block_new.copyTo(watermarked_image(cv::Range(x, x + block_size), cv::Range(y, y + block_size)));
        }

        // Compute difference and update watermarked image
        for (int i = 0; i < original_image.rows; ++i) {
            for (int j = 0; j < original_image.cols; ++j) {
                if (blank_image[i][j] > 0) {
                    int original_val = static_cast<int>(original_image.at<uchar>(i, j));
                    int watermarked_val = static_cast<int>(watermarked_image.at<uchar>(i, j));
                    int diff = (watermarked_val - original_val) * static_cast<int>(blank_image[i][j]);
                    int new_val = original_val + diff + static_cast<int>(std::round(blank_image[i][j]));
                    new_val = std::min(std::max(new_val, 0), 255);
                    watermarked_image.at<uchar>(i, j) = static_cast<uchar>(new_val);
                }
            }
        }

        // Compute wPSNR
        double w = wpsnr(original_image, watermarked_image, "utilities/csf.csv");
        std::cout << "[EMBEDDING] wPSNR: " << w << " dB" << std::endl;

        // Display watermarked image
        cv::imshow("Watermarked Image", watermarked_image);
        cv::waitKey(0);

        return watermarked_image;
    }

private:
    // Helper function to convert cv::Mat to vector of vector<float> and normalize
    std::vector<std::vector<float>> matToFloatVector(const cv::Mat& img) {
        std::vector<std::vector<float>> vec(img.rows, std::vector<float>(img.cols, 0.0f));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                vec[i][j] = static_cast<float>(img.at<uchar>(i, j)) / 255.0f;
            }
        }
        return vec;
    }

    // Helper function to subtract images
    std::vector<std::vector<float>> subtractImages(const std::vector<std::vector<float>>& img1, const cv::Mat& img2) {
        std::vector<std::vector<float>> difference(img1.size(), std::vector<float>(img1[0].size(), 0.0f));
        for (int i = 0; i < img1.size(); ++i) {
            for (int j = 0; j < img1[0].size(); ++j) {
                difference[i][j] = img1[i][j] - (static_cast<float>(img2.at<uchar>(i, j)) / 255.0f);
            }
        }
        return difference;
    }

    // Helper function to load CSV
    std::vector<std::vector<double>> loadCSV(const std::string& filename) {
        std::vector<std::vector<double>> data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open CSF file!" << std::endl;
            return data;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stod(value));
            }
            data.push_back(row);
        }
        file.close();
        return data;
    }

    // Helper function to perform 2D convolution
    std::vector<std::vector<double>> convolve2D(const std::vector<std::vector<float>>& img, const std::vector<std::vector<double>>& kernel) {
        int img_rows = img.size();
        int img_cols = img[0].size();
        int ker_rows = kernel.size();
        int ker_cols = kernel[0].size();
        int out_rows = img_rows - ker_rows + 1;
        int out_cols = img_cols - ker_cols + 1;

        if (out_rows <= 0 || out_cols <= 0) {
            std::cerr << "Error: Kernel size larger than image!" << std::endl;
            return std::vector<std::vector<double>>();
        }

        std::vector<std::vector<double>> output(out_rows, std::vector<double>(out_cols, 0.0));

        for (int i = 0; i < out_rows; ++i) {
            for (int j = 0; j < out_cols; ++j) {
                double sum = 0.0;
                for (int m = 0; m < ker_rows; ++m) {
                    for (int n = 0; n < ker_cols; ++n) {
                        sum += img[i + m][j + n] * kernel[m][n];
                    }
                }
                output[i][j] = sum;
            }
        }

        return output;
    }

    // Helper function to generate Gaussian kernel
    std::vector<std::vector<double>> generateGaussianKernel(int size, double sigma) {
        std::vector<std::vector<double>> kernel(size, std::vector<double>(size, 0.0));
        double mean = size / 2.0;
        double sum = 0.0; // For normalization

        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y) {
                kernel[x][y] = std::exp(-0.5 * (std::pow((x - mean) / sigma, 2.0) + std::pow((y - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);
                sum += kernel[x][y];
            }
        }

        // Normalize the kernel
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y) {
                kernel[x][y] /= sum;
            }
        }

        return kernel;
    }

    // Helper function to convert image to 2D vector
    std::vector<std::vector<unsigned char>> imageToVector(const cv::Mat& img) {
        std::vector<std::vector<unsigned char>> vec(img.rows, std::vector<unsigned char>(img.cols, 0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                vec[i][j] = img.at<uchar>(i, j);
            }
        }
        return vec;
    }

    // Helper function to accumulate differences into blank_image
    void accumulateDifferences(const std::vector<std::vector<unsigned char>>& attacked, const cv::Mat& original, std::vector<std::vector<double>>& blank_image) {
        for (int i = 0; i < original.rows; ++i) {
            for (int j = 0; j < original.cols; ++j) {
                blank_image[i][j] += std::abs(static_cast<int>(attacked[i][j]) - static_cast<int>(original.at<uchar>(i, j)));
            }
        }
    }

    // Helper function to get a block from the image
    std::vector<std::vector<unsigned char>> getBlock(const cv::Mat& img, int x, int y, int block_size) {
        std::vector<std::vector<unsigned char>> block(block_size, std::vector<unsigned char>(block_size, 0));
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                block[i][j] = img.at<uchar>(x + i, y + j);
            }
        }
        return block;
    }

    // Helper function to compute mean of a block
    double computeMean(const std::vector<std::vector<unsigned char>>& block) {
        double sum = 0.0;
        for (const auto& row : block) {
            for (auto val : row) {
                sum += val;
            }
        }
        return sum / (block.size() * block[0].size());
    }

    // Helper function to compute attack value
    double computeAttackValue(const std::vector<std::vector<double>>& blank_image, int x, int y, int block_size) {
        double sum = 0.0;
        for (int i = x; i < x + block_size; ++i) {
            for (int j = y; j < y + block_size; ++j) {
                sum += blank_image[i][j];
            }
        }
        return sum / (block_size * block_size);
    }

    // Helper function to get block as Eigen matrix
    Eigen::MatrixXd getBlockEigen(const cv::Mat& img, int x, int y, int block_size) {
        Eigen::MatrixXd mat(block_size, block_size);
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                mat(i, j) = static_cast<double>(img.at<uchar>(x + i, y + j));
            }
        }
        return mat;
    }

    // Helper function to apply median filter manually
    std::vector<std::vector<unsigned char>> applyMedianFilter(const cv::Mat& img, int kernel_size) {
        std::vector<std::vector<unsigned char>> median_filtered(img.rows, std::vector<unsigned char>(img.cols, 0));
        int pad = kernel_size / 2;

        // Pad the image with edge values
        std::vector<std::vector<unsigned char>> padded(img.rows + 2 * pad, std::vector<unsigned char>(img.cols + 2 * pad, 0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                padded[i + pad][j + pad] = img.at<uchar>(i, j);
            }
        }

        // Handle borders by replicating edge pixels
        for (int i = 0; i < padded.size(); ++i) {
            for (int j = 0; j < pad; ++j) {
                padded[i][j] = padded[i][pad];
                padded[i][padded[0].size() - 1 - j] = padded[i][padded[0].size() - 1 - pad];
            }
        }
        for (int j = 0; j < padded[0].size(); ++j) {
            for (int i = 0; i < pad; ++i) {
                padded[i][j] = padded[pad][j];
                padded[padded.size() - 1 - i][j] = padded[padded.size() - 1 - pad][j];
            }
        }

        // Apply median filter
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                std::vector<unsigned char> window;
                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        window.push_back(padded[i + m][j + n]);
                    }
                }
                std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
                median_filtered[i][j] = window[window.size() / 2];
            }
        }

        return median_filtered;
    }

    // Helper function to resize image manually using bilinear interpolation
    cv::Mat resizeImage(const cv::Mat& img, double scale) {
        int new_rows = static_cast<int>(img.rows * scale);
        int new_cols = static_cast<int>(img.cols * scale);
        std::vector<std::vector<unsigned char>> resized = bilinearResize(imageToVector(img), new_rows, new_cols);

        // If scaling up or down, resize back to original size
        std::vector<std::vector<unsigned char>> final_resized = bilinearResize(resized, img.rows, img.cols);

        // Convert back to cv::Mat
        cv::Mat resized_mat(img.rows, img.cols, CV_8UC1);
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                resized_mat.at<uchar>(i, j) = final_resized[i][j];
            }
        }
        return resized_mat;
    }

    // Helper function for bilinear resizing
    std::vector<std::vector<unsigned char>> bilinearResize(const std::vector<std::vector<unsigned char>>& img, int new_rows, int new_cols) {
        int orig_rows = img.size();
        int orig_cols = img[0].size();
        std::vector<std::vector<unsigned char>> resized(new_rows, std::vector<unsigned char>(new_cols, 0));

        double row_scale = static_cast<double>(orig_rows) / new_rows;
        double col_scale = static_cast<double>(orig_cols) / new_cols;

        for (int i = 0; i < new_rows; ++i) {
            double src_i = i * row_scale;
            int i_low = static_cast<int>(std::floor(src_i));
            int i_high = std::min(i_low + 1, orig_rows - 1);
            double delta_i = src_i - i_low;

            for (int j = 0; j < new_cols; ++j) {
                double src_j = j * col_scale;
                int j_low = static_cast<int>(std::floor(src_j));
                int j_high = std::min(j_low + 1, orig_cols - 1);
                double delta_j = src_j - j_low;

                double top = img[i_low][j_low] * (1 - delta_j) + img[i_low][j_high] * delta_j;
                double bottom = img[i_high][j_low] * (1 - delta_j) + img[i_high][j_high] * delta_j;
                double value = top * (1 - delta_i) + bottom * delta_i;

                resized[i][j] = static_cast<unsigned char>(std::round(std::min(std::max(value, 0.0), 255.0)));
            }
        }

        return resized;
    }

    // Helper function to perform 2D convolution on image with given kernel
    std::vector<std::vector<unsigned char>> convolveImage(const cv::Mat& img, const std::vector<std::vector<double>>& kernel) {
        int img_rows = img.rows;
        int img_cols = img.cols;
        int ker_rows = kernel.size();
        int ker_cols = kernel[0].size();
        int out_rows = img_rows - ker_rows + 1;
        int out_cols = img_cols - ker_cols + 1;

        if (out_rows <= 0 || out_cols <= 0) {
            std::cerr << "Error: Kernel size larger than image!" << std::endl;
            return std::vector<std::vector<unsigned char>>();
        }

        std::vector<std::vector<unsigned char>> output(img_rows, std::vector<unsigned char>(img_cols, 0));

        for (int i = 0; i < img_rows; ++i) {
            for (int j = 0; j < img_cols; ++j) {
                double sum = 0.0;
                for (int m = 0; m < ker_rows; ++m) {
                    for (int n = 0; n < ker_cols; ++n) {
                        int ii = i + m - ker_rows / 2;
                        int jj = j + n - ker_cols / 2;
                        if (ii >= 0 && ii < img_rows && jj >= 0 && jj < img_cols) {
                            sum += static_cast<double>(img.at<uchar>(ii, jj)) * kernel[m][n];
                        }
                    }
                }
                // Clamp the result
                sum = std::min(std::max(sum, 0.0), 255.0);
                output[i][j] = static_cast<unsigned char>(std::round(sum));
            }
        }

        return output;
    }

    // Helper function to perform bilinear resize (alternative)
    std::vector<std::vector<unsigned char>> bilinearResizeAlternative(const std::vector<std::vector<unsigned char>>& img, int new_rows, int new_cols) {
        // Implementation similar to the previous bilinearResize
        return bilinearResize(img, new_rows, new_cols);
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: watermark <original_image_path> <watermark_image_path>" << std::endl;
        return -1;
    }

    std::string original_image_path = argv[1];
    std::string watermark_image_path = argv[2];

    WatermarkEmbedder embedder;
    cv::Mat watermarked = embedder.embedding(original_image_path, watermark_image_path);

    // Save watermarked image
    cv::imwrite("watermarked_image.png", watermarked);
    std::cout << "Watermarked image saved as 'watermarked_image.png'" << std::endl;

    return 0;
}