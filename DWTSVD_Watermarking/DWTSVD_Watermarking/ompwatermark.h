#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <filesystem>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <omp.h>

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
vector<vector<double>> omp_createGaussianKernel(int kernelSize, double sigma);
Mat omp_apply_blur(const Mat& img, double sigma);

// Function to add Additive White Gaussian Noise (AWGN)
Mat omp_apply_awgn(const Mat& img, double stddev);

// Function to apply median filtering
Mat omp_apply_median_filter(const Mat& img, int kernel_size);

// Function to apply sharpening
Mat omp_apply_sharpen(const Mat& img, double sigma, double alpha);

// Function to resize image (downscale and upscale)
double omp_bilinearInterpolate(const Mat& img, double x, double y);
Mat omp_resizeImage(const Mat& img, int newRows, int newCols);
Mat omp_apply_resize(const Mat& img, double scale);

// Function to perform Haar Wavelet Transform (DWT)
void omp_haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH);

// Function to perform Inverse Haar Wavelet Transform (IDWT)
void omp_inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst);

bool omp_is_image_size_sufficient(int original_width, int original_height, int watermark_width, int watermark_height, int block_size, int n_blocks_to_embed);

void omp_compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt);

void omp_reconstruct_matrix(const Mat& U, const Mat& S, const Mat& Vt, Mat& reconstructed);

cv::Mat omp_customConvert8U(const cv::Mat& src);

cv::Mat omp_extractPrecisionDifference(const Mat& mat8u, const Mat& mat64f);

cv::Mat omp_combineMatPrecision(const cv::Mat& integerMat, const cv::Mat& precisionMat);

// Function to save the singular value matrix (S) as a secret key
void omp_save_singular_values(const Mat& S, const string& key_file);

// Function to load the singular value matrix (S) from the secret key file
Mat omp_load_singular_values(const string& key_file);

void omp_save_selected_blocks(const vector<Block>& selected_blocks, const string& key_file);

vector<Block> omp_load_selected_blocks(const string& key_file);

void omp_save_svd_components(const Mat& U, const Mat& S, const Mat& Vt, const string& filename);
void omp_load_svd_components(const string& filename, Mat& U, Mat& S, Mat& Vt);

void omp_saveMatVectorToFile(const std::vector<Mat>& mat_vector, const std::string& filename);

std::vector<Mat> omp_loadMatVectorFromFile(const std::string& filename);

void omp_savePrecisionMat(const cv::Mat& precisionMat, const std::string& filename);

cv::Mat omp_loadPrecisionMat(const std::string& filename);

// Function to embed watermark
Mat omp_embed_watermark(
    const Mat& original, const Mat& watermark, double alpha,
    const string& key_filename, int wm_width = 32, int wm_height = 32,
    int n_blocks_to_embed = 32, int block_size = 4, double spatial_weight = 0.33
);

Mat omp_extract_watermark(const Mat& watermarked_int_image, const string& key_filename, int n_blocks_to_extract = 32, int block_size = 4, double alpha = 5.11);

int omp(std::chrono::milliseconds* execution_time, double* psnr, bool isDisplay = false, string original_image_path = "home.jpg", string watermark_image_path = "mono.png", int watermark_width = 64, int watermark_height = 64);

//int sequential(std::chrono::milliseconds* embed_time, std::chrono::milliseconds* extract_time, bool isDisplay = false, string original_image_path = "home.jpg", string watermark_image_path = "mono.png");