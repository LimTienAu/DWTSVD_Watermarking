#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

// Function to apply Gaussian blur
Mat blur(const Mat& img, double sigma) {
    Mat blurred;
    GaussianBlur(img, blurred, Size(0, 0), sigma);
    return blurred;
}

// Function to add AWGN (Additive White Gaussian Noise)
Mat awgn(const Mat& img, double stddev) {
    Mat noise = Mat(img.size(), CV_64F);
    randn(noise, 0, stddev);
    Mat noisy_img;
    img.convertTo(noisy_img, CV_64F);
    noisy_img += noise;
    noisy_img = cv::max(0, cv::min(255, noisy_img)); // Clip to 0-255
    noisy_img.convertTo(noisy_img, CV_8U);
    return noisy_img;
}

// Function to apply median filtering
Mat median_filter(const Mat& img, int kernel_size) {
    Mat filtered;
    medianBlur(img, filtered, kernel_size);
    return filtered;
}

// Function to apply sharpening
Mat sharpen(const Mat& img, double sigma, double alpha) {
    Mat blurred, sharpened;
    GaussianBlur(img, blurred, Size(0, 0), sigma);
    sharpened = img + alpha * (img - blurred);
    return sharpened;
}

// Function to resize image (downscale and upscale)
Mat resize_image(const Mat& img, double scale) {
    Mat resized, restored;
    resize(img, resized, Size(), scale, scale, INTER_LINEAR);
    resize(resized, restored, img.size(), 0, 0, INTER_LINEAR);
    return restored;
}

// DWT (Haar Wavelet Transform) helper functions
void haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH) {
    int rows = src.rows / 2;
    int cols = src.cols / 2;

    LL = Mat(rows, cols, CV_64F);
    LH = Mat(rows, cols, CV_64F);
    HL = Mat(rows, cols, CV_64F);
    HH = Mat(rows, cols, CV_64F);

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

void inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst) {
    int rows = LL.rows;
    int cols = LL.cols;

    dst = Mat(rows * 2, cols * 2, CV_64F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
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
    dst.convertTo(dst, CV_8U);
}

// SVD helper function
void compute_svd(const Mat& src, Mat& U, Mat& S, Mat& V) {
    SVD::compute(src, S, U, V);
}

// Main watermark embedding function
Mat embed_watermark(const Mat& original, const Mat& watermark, double alpha) {
    // Prepare the watermarked image
    Mat watermarked_image = original.clone();
    watermarked_image.convertTo(watermarked_image, CV_64F);

    int block_size = 4;
    vector<Rect> blocks;

    // Collect blocks from the image
    for (int i = 0; i < original.rows; i += block_size) {
        for (int j = 0; j < original.cols; j += block_size) {
            blocks.push_back(Rect(j, i, block_size, block_size));
        }
    }

    // Sort blocks by mean intensity for selection
    sort(blocks.begin(), blocks.end(), [&](const Rect& a, const Rect& b) {
        return mean(original(a))[0] > mean(original(b))[0];
        });

    // Embed watermark in the selected blocks
    for (int idx = 0; idx < min(32, (int)blocks.size()); ++idx) {
        Rect block = blocks[idx];
        Mat sub_img = original(block);

        // Perform DWT on the block
        Mat LL, LH, HL, HH;
        haar_wavelet_transform(sub_img, LL, LH, HL, HH);

        // SVD on the LL sub-band
        Mat U, S, V;
        compute_svd(LL, U, S, V);

        // Modify singular values with watermark
        for (int i = 0; i < watermark.rows && i < S.rows; ++i) {
            S.at<double>(i, i) += alpha * watermark.at<double>(idx % 32, i);
        }

        // Reconstruct LL sub-band
        Mat modified_LL;
        modified_LL = U * Mat::diag(S) * V;

        // Perform inverse DWT
        inverse_haar_wavelet_transform(modified_LL, LH, HL, HH, sub_img);
        sub_img.copyTo(watermarked_image(block));
    }

    watermarked_image.convertTo(watermarked_image, CV_8U);
    return watermarked_image;
}

int main() {

    // Example input image and watermark
    Mat original_image = imread("C:\\Users\\lim\\OneDrive\\Pictures\\persona.jpg", IMREAD_GRAYSCALE);
    if (original_image.empty()) {
        cerr << "Error: Could not load input image." << endl;
        return -1;
    }

    Mat watermark = imread("C:\\Users\\lim\\OneDrive\\Pictures\\persona.jpg", IMREAD_GRAYSCALE);
    if (watermark.empty()) {
        cerr << "Error: Could not load watermark image." << endl;
        return -1;
    }

    // Resize watermark to 32x32
    resize(watermark, watermark, Size(32, 32), 0, 0, INTER_LINEAR);
    watermark.convertTo(watermark, CV_64F, 1.0 / 255.0); // Normalize to [0, 1]

    //// Embed watermark
    Mat watermarked_image = embed_watermark(original_image, watermark, 5.11);

    // Save and display result
    imwrite("watermarked_image.png", watermarked_image);
    imshow("Watermarked Image", watermarked_image);
    waitKey(0);

    return 0;
}
