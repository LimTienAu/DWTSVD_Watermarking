#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

// Structure to hold block information
struct Block {
    Rect location;
    double spatial_value;
    double attack_value;
    double merit;
};

// Function to apply Gaussian blur
Mat apply_blur(const Mat& img, double sigma) {
    Mat blurred;
    GaussianBlur(img, blurred, Size(0, 0), sigma);
    return blurred;
}

// Function to add Additive White Gaussian Noise (AWGN)
Mat apply_awgn(const Mat& img, double stddev) {
    Mat noise = Mat(img.size(), CV_64F);
    randn(noise, 0, stddev);
    Mat noisy_img;
    img.convertTo(noisy_img, CV_64F);
    noisy_img += noise;
    noisy_img = max(noisy_img, 0.0);
    noisy_img = min(noisy_img, 255.0);
    noisy_img.convertTo(noisy_img, CV_8U);
    return noisy_img;
}

// Function to apply median filtering
Mat apply_median_filter(const Mat& img, int kernel_size) {
    Mat filtered;
    medianBlur(img, filtered, kernel_size);
    return filtered;
}

// Function to apply sharpening
Mat apply_sharpen(const Mat& img, double sigma, double alpha) {
    Mat blurred, sharpened;
    GaussianBlur(img, blurred, Size(0, 0), sigma);
    sharpened = img + alpha * (img - blurred);
    sharpened = max(sharpened, 0.0);
    sharpened = min(sharpened, 255.0);
    sharpened.convertTo(sharpened, CV_8U);
    return sharpened;
}

// Function to resize image (downscale and upscale)
Mat apply_resize(const Mat& img, double scale) {
    Mat resized, restored;
    resize(img, resized, Size(), scale, scale, INTER_LINEAR);
    resize(resized, restored, img.size(), 0, 0, INTER_LINEAR);
    return restored;
}

// Function to perform Haar Wavelet Transform (DWT)
void haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH) {
    int rows = src.rows / 2;
    int cols = src.cols / 2;

    LL = Mat(rows, cols, CV_64F, Scalar(0));
    LH = Mat(rows, cols, CV_64F, Scalar(0));
    HL = Mat(rows, cols, CV_64F, Scalar(0));
    HH = Mat(rows, cols, CV_64F, Scalar(0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double a = src.at<uchar>(i * 2, j * 2);
            double b = src.at<uchar>(i * 2, j * 2 + 1);
            double c = src.at<uchar>(i * 2 + 1, j * 2);
            double d = src.at<uchar>(i * 2 + 1, j * 2 + 1);

            LL.at<double>(i, j) = (a + b + c + d) / 4.0;
            LH.at<double>(i, j) = (a - b + c - d) / 4.0;
            HL.at<double>(i, j) = (a + b - c - d) / 4.0;
            HH.at<double>(i, j) = (a - b - c + d) / 4.0;
        }
    }
}

// Function to perform Inverse Haar Wavelet Transform (IDWT)
void inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst) {
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
    normalize(dst, dst, 0, 255, NORM_MINMAX);
}

// Function to compute SVD
void compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt) {
    Mat src_double;
    src.convertTo(src_double, CV_64F);
    SVD svd(src_double, SVD::FULL_UV);
    U = svd.u;
    S = svd.w;
    Vt = svd.vt;
}

// Function to embed watermark
Mat embed_watermark(const Mat& original, const Mat& watermark, double alpha, int n_blocks_to_embed = 32, int block_size = 4, double spatial_weight = 0.33) {
    // Initialize variables
    Mat watermarked_image = original.clone();
    watermarked_image.convertTo(watermarked_image, CV_64F);

    // Initialize blank_image
    Mat blank_image = Mat::zeros(original.size(), CV_64F);

    // Apply various attacks and accumulate differences
    // 1. Gaussian Blur
    vector<double> blur_sigma_values = { 0.1, 0.5, 1, 2, 1.0, 2.0 };
    for (auto sigma : blur_sigma_values) {
        Mat attacked = apply_blur(original, sigma);
        Mat diff;
        absdiff(attacked, original, diff);
        diff.convertTo(diff, CV_64F);
        blank_image += diff;
    }

    // 2. Median Filtering
    vector<int> median_kernel_sizes = { 3, 5, 7, 9, 11 };
    for (auto k : median_kernel_sizes) {
        Mat attacked = apply_median_filter(original, k);
        Mat diff;
        absdiff(attacked, original, diff);
        diff.convertTo(diff, CV_64F);
        blank_image += diff;
    }

    // 3. Additive White Gaussian Noise
    vector<double> awgn_std_values = { 0.1, 0.5, 2, 5, 10 };
    for (auto stddev : awgn_std_values) {
        Mat attacked = apply_awgn(original, stddev);
        Mat diff;
        absdiff(attacked, original, diff);
        diff.convertTo(diff, CV_64F);
        blank_image += diff;
    }

    // 4. Sharpening
    vector<double> sharpen_sigma_values = { 0.1, 0.5, 2, 100 };
    vector<double> sharpen_alpha_values = { 0.1, 0.5, 1, 2 };
    for (auto sigma : sharpen_sigma_values) {
        for (auto a : sharpen_alpha_values) {
            Mat attacked = apply_sharpen(original, sigma, a);
            Mat diff;
            absdiff(attacked, original, diff);
            diff.convertTo(diff, CV_64F);
            blank_image += diff;
        }
    }

    // 5. Resizing
    vector<double> resize_scale_values = { 0.5, 0.75, 0.9, 1.1, 1.5 };
    for (auto scale : resize_scale_values) {
        Mat attacked = apply_resize(original, scale);
        Mat diff;
        absdiff(attacked, original, diff);
        diff.convertTo(diff, CV_64F);
        blank_image += diff;
    }

    // Block selection based on spatial and attack values
    vector<Block> blocks_to_watermark;

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
                blocks_to_watermark.push_back(blk);
            }
        }
    }

    // Sort blocks based on spatial value (descending)
    sort(blocks_to_watermark.begin(), blocks_to_watermark.end(), [&](const Block& a, const Block& b) {
        return a.spatial_value > b.spatial_value;
        });

    // Assign merit based on spatial rank
    for (int i = 0; i < blocks_to_watermark.size(); ++i) {
        blocks_to_watermark[i].merit += i * spatial_weight;
    }

    // Sort blocks based on attack value (ascending)
    sort(blocks_to_watermark.begin(), blocks_to_watermark.end(), [&](const Block& a, const Block& b) {
        return a.attack_value < b.attack_value;
        });

    // Assign merit based on attack rank
    double attack_weight = 1.0 - spatial_weight;
    for (int i = 0; i < blocks_to_watermark.size(); ++i) {
        blocks_to_watermark[i].merit += i * attack_weight;
    }

    // Sort blocks based on total merit (descending)
    sort(blocks_to_watermark.begin(), blocks_to_watermark.end(), [&](const Block& a, const Block& b) {
        return a.merit > b.merit;
        });

    // Select top n_blocks_to_embed blocks
    vector<Block> selected_blocks;
    for (int i = 0; i < min(n_blocks_to_embed, (int)blocks_to_watermark.size()); ++i) {
        selected_blocks.push_back(blocks_to_watermark[i]);
    }

    // Precompute SVD of the watermark
    Mat watermark_resized;
    resize(watermark, watermark_resized, Size(32, 32), 0, 0, INTER_LINEAR);
    watermark_resized.convertTo(watermark_resized, CV_64F, 1.0 / 255.0); // Normalize

    Mat Uwm, Swm, Vtwm;
    compute_svd(watermark_resized, Uwm, Swm, Vtwm);

    // Embed watermark into selected blocks
    for (int idx = 0; idx < selected_blocks.size(); ++idx) {
        Rect block_loc = selected_blocks[idx].location;
        Mat block = watermarked_image(block_loc).clone();

        // Perform DWT on the block
        Mat LL, LH, HL, HH;
        haar_wavelet_transform(block, LL, LH, HL, HH);

        // Perform SVD on LL subband
        Mat Uc, Sc, Vtc;
        compute_svd(LL, Uc, Sc, Vtc);

        // Modify singular values
        // Ensure we don't exceed the size of Swm
        int swm_size = Swm.rows;
        for (int i = 0; i < min((int)Sc.rows, (int)swm_size); ++i) {
            Sc.at<double>(i) += alpha * Swm.at<double>(idx % Swm.rows, i);
        }

        // Reconstruct LL subband
        Mat modified_S = Mat::diag(Sc);
        Mat modified_LL = Uc * modified_S * Vtc;

        // Perform inverse DWT
        Mat reconstructed_block;
        inverse_haar_wavelet_transform(modified_LL, LH, HL, HH, reconstructed_block);

        // Replace the block in the watermarked image
        watermarked_image(block_loc) = reconstructed_block;
    }

    // Finalize watermarked image
    watermarked_image = watermarked_image + blank_image;
    watermarked_image = min(watermarked_image, 255.0);
    watermarked_image = max(watermarked_image, 0.0);
    watermarked_image.convertTo(watermarked_image, CV_8U);

    return watermarked_image;
}

#include <filesystem>
int main() {
    std::string original_image_path = "mono.png";
    if (!std::filesystem::exists(original_image_path)) {
        std::cerr << "File does not exist: " << original_image_path << std::endl;
        return -1;
    }
    // ============================ //
    //      Fixed Image Paths       //
    // ============================ //

    // Define the paths to the original and watermark images
    // You can use absolute paths or relative paths based on your project structure
    // Example using absolute paths:
    // string original_image_path = "C:\\Users\\YourUsername\\Pictures\\original.jpg";
    // string watermark_image_path = "C:\\Users\\YourUsername\\Pictures\\watermark.png";

    // Example using relative paths (assuming images are in the project directory)
    
    string watermark_image_path = "kk.jpg";

    // ============================ //
    //      Load Original Image     //
    // ============================ //

    // Load original image in grayscale
    Mat original_image = imread(original_image_path, IMREAD_GRAYSCALE);
    if (original_image.empty()) {
        cerr << "Error: Could not load original image from path: " << original_image_path << endl;
        return -1;
    }

    // Resize original image to 512x512 if not already
    if (original_image.rows != 512 || original_image.cols != 512) {
        resize(original_image, original_image, Size(512, 512), 0, 0, INTER_LINEAR);
        cout << "Original image resized to 512x512." << endl;
    }

    // Display original image
    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", original_image);
    // Wait for a key press to proceed
    waitKey(0);

    // ============================ //
    //      Load Watermark Image    //
    // ============================ //

    // Load watermark image in grayscale
    Mat watermark_image = imread(watermark_image_path, IMREAD_GRAYSCALE);
    if (watermark_image.empty()) {
        cerr << "Error: Could not load watermark image from path: " << watermark_image_path << endl;
        return -1;
    }

    // Display watermark image
    namedWindow("Watermark Image", WINDOW_NORMAL);
    imshow("Watermark Image", watermark_image);
    // Wait for a key press to proceed
    waitKey(0);

    // ============================ //
    //      Embed Watermark         //
    // ============================ //

    double alpha = 5.11; // Embedding strength
    Mat watermarked_image = embed_watermark(original_image, watermark_image, alpha);

    // ============================ //
    //      Display Results         //
    // ============================ //

    // Display watermarked image
    namedWindow("Watermarked Image", WINDOW_NORMAL);
    imshow("Watermarked Image", watermarked_image);
    // Wait for a key press to proceed
    waitKey(0);

    // Save watermarked image
    string output_image_path = "watermarked_image.png";
    bool isSaved = imwrite(output_image_path, watermarked_image);
    if (isSaved) {
        cout << "Watermarked image saved as '" << output_image_path << "'." << endl;
    }
    else {
        cerr << "Error: Could not save watermarked image." << endl;
    }

    return 0;
}
