#include "Common_include.h"

//void cuda_gaussian_blur(const cv::Mat& input, double sigma, cv::Mat& diff_accum);
//void cuda_median_filter(const cv::Mat& input, int ksize, cv::Mat& diff_accum);
//void cuda_process_multiple_kernels(const cv::Mat& input, const std::vector<int>& kernel_sizes, cv::Mat& diff_accum);
void processAllOperations(const cv::Mat& input, cv::Mat& diff_accum,
    const std::vector<double>& blur_sigmas,
    const std::vector<int>& median_kernels,
    const std::vector<double>& noise_stddevs);