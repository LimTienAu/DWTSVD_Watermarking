#pragma once
#include "Common_include.h"

using namespace cv;
using namespace std;


// Function to perform Haar Wavelet Transform (DWT)
void cuda_haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH);

// Function to perform Inverse Haar Wavelet Transform (IDWT)
void cuda_inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst);

bool cuda_is_image_size_sufficient(int original_width, int original_height, int watermark_width, int watermark_height, int block_size, int n_blocks_to_embed);

void cuda_compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt);

void cuda_reconstruct_matrix(const Mat& U, const Mat& S, const Mat& Vt, Mat& reconstructed);

cv::Mat cuda_customConvert8U(const cv::Mat& src);

cv::Mat cuda_extractPrecisionDifference(const Mat& mat8u, const Mat& mat64f);

cv::Mat cuda_combineMatPrecision(const cv::Mat& integerMat, const cv::Mat& precisionMat);

// Function to save the singular value matrix (S) as a secret key
void cuda_save_singular_values(const Mat& S, const string& key_file);

// Function to load the singular value matrix (S) from the secret key file
Mat cuda_load_singular_values(const string& key_file);

void cuda_save_selected_blocks(const vector<Block>& selected_blocks, const string& key_file);

vector<Block> cuda_load_selected_blocks(const string& key_file);

void cuda_save_svd_components(const Mat& U, const Mat& S, const Mat& Vt, const string& filename);
void cuda_load_svd_components(const string& filename, Mat& U, Mat& S, Mat& Vt);

void cuda_saveMatVectorToFile(const std::vector<Mat>& mat_vector, const std::string& filename);

std::vector<Mat> cuda_loadMatVectorFromFile(const std::string& filename);

void cuda_savePrecisionMat(const cv::Mat& precisionMat, const std::string& filename);

cv::Mat cuda_loadPrecisionMat(const std::string& filename);

// Function to embed watermark
Mat cuda_embed_watermark(
    const Mat& original, const Mat& watermark, double alpha,
    const string& key_filename, int wm_width = 32, int wm_height = 32,
    int n_blocks_to_embed = 32, int block_size = 4, double spatial_weight = 0.33
);

Mat cuda_extract_watermark(const Mat& watermarked_int_image, const string& key_filename, int n_blocks_to_extract = 32, int block_size = 4, double alpha = 5.11);

int cuda_main(std::chrono::milliseconds* execution_time, double* psnr, bool isDisplay = false, string original_image_path = "home.jpg", string watermark_image_path = "mono.png", int watermark_width=64, int watermark_height=64);
