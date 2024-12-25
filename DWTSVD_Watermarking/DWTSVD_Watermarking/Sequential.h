#pragma once
#include "Common_include.h"
#include <math.h>

using namespace cv;
using namespace std;

// Function to apply Gaussian blur
Mat apply_blur(const Mat& img, double sigma);

// Function to add Additive White Gaussian Noise (AWGN)
Mat apply_awgn(const Mat& img, double stddev);

// Function to apply median filtering
Mat apply_median_filter(const Mat& img, int kernel_size);

// Function to apply sharpening
Mat apply_sharpen(const Mat& img, double sigma, double alpha);

// Function to resize image (downscale and upscale)
Mat apply_resize(const Mat& img, double scale);

// Function to perform Haar Wavelet Transform (DWT)
void haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH);

// Function to perform Inverse Haar Wavelet Transform (IDWT)
void inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst);

bool is_image_size_sufficient(int original_width, int original_height, int watermark_width, int watermark_height, int block_size, int n_blocks_to_embed);

void compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt);

void reconstruct_matrix(const Mat& U, const Mat& S, const Mat& Vt, Mat& reconstructed);

cv::Mat customConvert8U(const cv::Mat& src);

cv::Mat extractPrecisionDifference(const Mat& mat8u, const Mat& mat64f);

cv::Mat combineMatPrecision(const cv::Mat& integerMat, const cv::Mat& precisionMat);

// Function to save the singular value matrix (S) as a secret key
void save_singular_values(const Mat& S, const string& key_file);

// Function to load the singular value matrix (S) from the secret key file
Mat load_singular_values(const string& key_file);

void save_selected_blocks(const vector<Block>& selected_blocks, const string& key_file);

vector<Block> load_selected_blocks(const string& key_file);

void save_svd_components(const Mat& U, const Mat& S, const Mat& Vt, const string& filename);
void load_svd_components(const string& filename, Mat& U, Mat& S, Mat& Vt);

void saveMatVectorToFile(const std::vector<Mat>& mat_vector, const std::string& filename);

std::vector<Mat> loadMatVectorFromFile(const std::string& filename);

void savePrecisionMat(const cv::Mat& precisionMat, const std::string& filename);

cv::Mat loadPrecisionMat(const std::string& filename);

// Function to embed watermark
Mat embed_watermark(
    const Mat& original, const Mat& watermark, double alpha,
    const string& key_filename, int wm_width = 32, int wm_height = 32,
    int n_blocks_to_embed = 32, int block_size = 4, double spatial_weight = 0.33, std::chrono::milliseconds** execution_time = 0
);

Mat extract_watermark(const Mat& watermarked_int_image, const string& key_filename, int n_blocks_to_extract = 32, int block_size = 4, double alpha = 5.11);

int sequential(std::chrono::milliseconds* execution_time, double* psnr, bool isDisplay = false, string original_image_path = "home.jpg", string watermark_image_path = "mono.png");
