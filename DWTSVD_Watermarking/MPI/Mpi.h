#pragma once
#include "Common_include.h"


using namespace cv;
using namespace std;


////Function to apply Gaussian blur
//Mat mpi_apply_blur(const Mat& img, double sigma);
//
//// Function to add Additive White Gaussian Noise (AWGN)
//Mat mpi_apply_awgn(const Mat& img, double stddev);
//
//// Function to apply median filtering
//Mat mpi_apply_median_filter(const Mat& img, int kernel_size);
//
//// Function to apply sharpening
//Mat mpi_apply_sharpen(const Mat& img, double sigma, double alpha);
//
//// Function to resize image (downscale and upscale)
//Mat mpi_apply_resize(const Mat& img, double scale);


vector<vector<double>> mpi_createGaussianKernel(int kernelSize, double sigma);

Mat mpi_apply_blur(const Mat& img, double sigma, int rank, int size);

Mat mpi_apply_awgn(const Mat& img, double stddev, int rank, int size);

Mat mpi_apply_median_filter(const Mat& img, int kernel_size, int rank, int size);

Mat mpi_apply_sharpen(const Mat& img, double sigma, double alpha);

double mpi_bilinearInterpolate(const Mat& img, double x, double y);

Mat mpi_resizeImage(const Mat& img, int newRows, int newCols);

Mat mpi_apply_resize(const Mat& img, double scale);

// Function to perform Haar Wavelet Transform (DWT)
void mpi_haar_wavelet_transform(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH);

// Function to perform Inverse Haar Wavelet Transform (IDWT)
void mpi_inverse_haar_wavelet_transform(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst);

bool mpi_is_image_size_sufficient(int original_width, int original_height, int watermark_width, int watermark_height, int block_size, int n_blocks_to_embed);

void mpi_compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt);

void mpi_reconstruct_matrix(const Mat& U, const Mat& S, const Mat& Vt, Mat& reconstructed);

cv::Mat mpi_customConvert8U(const cv::Mat& src);

cv::Mat mpi_extractPrecisionDifference(const Mat& mat8u, const Mat& mat64f);

cv::Mat mpi_combineMatPrecision(const cv::Mat& integerMat, const cv::Mat& precisionMat);

// Function to save the singular value matrix (S) as a secret key
void mpi_save_singular_values(const Mat& S, const string& key_file);

// Function to load the singular value matrix (S) from the secret key file
Mat mpi_load_singular_values(const string& key_file);

void mpi_save_selected_blocks(const vector<Block>& selected_blocks, const string& key_file);

vector<Block> mpi_load_selected_blocks(const string& key_file);

void mpi_save_svd_components(const Mat& U, const Mat& S, const Mat& Vt, const string& filename);
void mpi_load_svd_components(const string& filename, Mat& U, Mat& S, Mat& Vt);

void mpi_saveMatVectorToFile(const std::vector<Mat>& mat_vector, const std::string& filename);

std::vector<Mat> mpi_loadMatVectorFromFile(const std::string& filename);

void mpi_savePrecisionMat(const cv::Mat& precisionMat, const std::string& filename);

cv::Mat mpi_loadPrecisionMat(const std::string& filename);


void mpi_broadcast_image(Mat& img, int root, MPI_Comm comm, int rank);

// Function to embed watermark
Mat mpi_embed_watermark(std::chrono::milliseconds** time,
    const Mat& original, const Mat& watermark, double alpha,
    const string& key_filename, int wm_width = 32, int wm_height = 32,
    int n_blocks_to_embed = 32, int block_size = 4, double spatial_weight = 0.33, int rank = 0, int size = 0
);

Mat mpi_extract_watermark(const Mat& watermarked_int_image, const string& key_filename, int n_blocks_to_extract = 32, int block_size = 4, double alpha = 5.11);

void mpi_parallel_optimization(
    const Mat& original, Mat& blank_image,
    int block_size, double spatial_weight,
    int n_blocks_to_embed, vector<Block>& selected_blocks,
    int rank=0, int size=0);

//void mpi_optimized_processing(const Mat& original, Mat& blank_image, int block_size, double spatial_weight, int n_blocks_to_embed, vector<Block>& selected_blocks);

int mpi(std::chrono::milliseconds* time, double* psnr, bool isDisplay = false, string original_image_path = "home.jpg", string watermark_image_path = "mono.png", int rank=0, int size=0);



//to go in rank 0