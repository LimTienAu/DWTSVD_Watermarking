#include "Sequential.h"

// Function to apply Gaussian blur
Mat apply_blur(const Mat& img, double sigma) {
    Mat blurred;
    cv::GaussianBlur(img, blurred, Size(0, 0), sigma);
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
    //normalize(dst, dst, 0, 255, NORM_MINMAX);
}

bool is_image_size_sufficient(int original_width, int original_height, int watermark_width, int watermark_height, int block_size, int n_blocks_to_embed) {
    // Calculate the total required blocks for the watermark
    int total_required_blocks = ceil((double)(watermark_width * watermark_height) / (block_size * block_size));

    // Calculate the total available blocks in the original image
    int total_image_blocks = (original_width / block_size) * (original_height / block_size);

    // Use the minimum of available blocks and n_blocks_to_embed
    int available_blocks = min(total_image_blocks, n_blocks_to_embed);

    // Check if available blocks are sufficient
    return available_blocks >= total_required_blocks;
}

void compute_svd(const Mat& src, Mat& U, Mat& S, Mat& Vt) {
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

void reconstruct_matrix(const Mat& U, const Mat& S, const Mat& Vt, Mat& reconstructed) {
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

cv::Mat customConvert8U(const cv::Mat& src) {
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

cv::Mat extractPrecisionDifference(const Mat& mat8u, const Mat& mat64f) {
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

cv::Mat combineMatPrecision(const cv::Mat& integerMat, const cv::Mat& precisionMat) {
    cv::Mat dst(precisionMat.rows, precisionMat.cols, CV_64F);
    for (int i = 0; i < precisionMat.rows; ++i) {
        for (int j = 0; j < precisionMat.cols; ++j) {
            dst.at<double>(i, j) = precisionMat.at<double>(i, j) + (integerMat.at<uchar>(i, j));
        }
    }

    return dst;
}

// Function to save the singular value matrix (S) as a secret key
void save_singular_values(const Mat& S, const string& key_file) {
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
Mat load_singular_values(const string& key_file) {
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

void save_selected_blocks(const vector<Block>& selected_blocks, const string& key_file) {
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
        //std::cout << "Selected blocks information saved to " << key_file << std::endl;
    }
    else {
        std::cerr << "Error: Could not save selected blocks to file." << std::endl;
    }
}

vector<Block> load_selected_blocks(const string& key_file) {
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

void save_svd_components(const Mat& U, const Mat& S, const Mat& Vt, const string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "U" << U;
    fs << "S" << S;
    fs << "Vt" << Vt;

    fs.release();
}

void load_svd_components(const string& filename, Mat& U, Mat& S, Mat& Vt) {
    FileStorage fs(filename, FileStorage::READ);

    fs["U"] >> U;
    fs["S"] >> S;
    fs["Vt"] >> Vt;

    fs.release();
}

void saveMatVectorToFile(const std::vector<Mat>& mat_vector, const std::string& filename) {
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "MatVector" << "[";
    for (const auto& mat : mat_vector) {
        fs << mat;
    }
    fs << "]";

    fs.release();
}

std::vector<Mat> loadMatVectorFromFile(const std::string& filename) {
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

void savePrecisionMat(const cv::Mat& precisionMat, const std::string& filename) {
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

cv::Mat loadPrecisionMat(const std::string& filename) {
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
Mat embed_watermark(
    const Mat& original, const Mat& watermark, double alpha, 
    const string& key_filename, int wm_width, int wm_height,
    int n_blocks_to_embed, int block_size, double spatial_weight, std::chrono::milliseconds** execution_time
) {
    // Initialize variables
    Mat watermarked_image = original.clone();
    watermarked_image.convertTo(watermarked_image, CV_64F);

    // Initialize blank_image
    Mat blank_image = Mat::zeros(original.size(), CV_64F);

    auto embed_begin = std::chrono::high_resolution_clock::now();
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

    auto embed_end = std::chrono::high_resolution_clock::now();
    **execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(embed_end - embed_begin);

    // Precompute SVD of the watermark
    wm_width = min(watermark.cols, wm_width); // Set max watermark size dynamically
    wm_height = min(watermark.rows, wm_height);
    Mat watermark_resized;
    resize(watermark, watermark_resized, Size(wm_width, wm_height), 0, 0, INTER_LINEAR);
    watermark_resized.convertTo(watermark_resized, CV_64F, 1.0 / 255.0); // Normalize

    //Save resized watermark image
    Mat save_watermark = watermark_resized.clone();
    normalize(save_watermark, save_watermark, 0, 255, NORM_MINMAX);
    save_watermark.convertTo(save_watermark, CV_8U);
    imwrite(key_filename + "_actualwatermark.tiff", save_watermark);

    Mat Uwm, Swm, Vtwm;
    compute_svd(watermark_resized, Uwm, Swm, Vtwm);
    save_svd_components(Uwm.clone(), Swm.clone(), Vtwm.clone(), key_filename + "wm_svd");
    save_selected_blocks(selected_blocks, key_filename + "_block");

    vector<Mat> original_Sc;
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
        
        original_Sc.push_back(Sc.clone());

        // Embed watermark subset
        int shape_LL_tmp = Sc.rows; // Assuming rows represent shape_LL_tmp
        int start_idx = (idx * shape_LL_tmp) % Swm.rows; // Modulo ensures cyclic access to Swm
        int end_idx = start_idx + shape_LL_tmp;

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
        reconstruct_matrix(Uc, Sc, Vtc, modified_LL);
        
        inverse_haar_wavelet_transform(modified_LL, LH, HL, HH, reconstructed_block);
        // Replace the block in the watermarked image
        reconstructed_block.copyTo(watermarked_image(block_loc));
    }
    
    saveMatVectorToFile(original_Sc, key_filename + "ori_s");
    Mat watermarked_image_int;
    watermarked_image.convertTo(watermarked_image_int, CV_8U);
    Mat precision_obj = extractPrecisionDifference(watermarked_image_int, watermarked_image);
    savePrecisionMat(precision_obj, key_filename + "_precision");

    return watermarked_image_int;
}

Mat extract_watermark(const Mat& watermarked_int_image, const string& key_filename, int n_blocks_to_extract, int block_size, double alpha) {
    // Load singular values and selected blocks
    Mat Uwm, Swm, Vtwm;
    vector<Block> selected_blocks = load_selected_blocks(key_filename + "_block");
    load_svd_components(key_filename +"wm_svd", Uwm, Swm, Vtwm);
    vector<Mat> ori_S  = loadMatVectorFromFile( key_filename + "ori_s"); 
    Mat precision = loadPrecisionMat(key_filename + "_precision");
    Mat watermarked_image = combineMatPrecision(watermarked_int_image, precision);

    // Initialize watermark image
    Mat extracted_watermark_S = Mat::zeros(Swm.size(), CV_64F);
    // Iterate over the selected blocks to extract watermark components
    for (int idx = 0; idx < min(n_blocks_to_extract, (int)selected_blocks.size()); ++idx) {
        Rect block_loc = selected_blocks[idx].location;
        Mat block = watermarked_image(block_loc).clone();
        block.convertTo(block, CV_64F);
        
        // Perform DWT on the block
        Mat LL, LH, HL, HH;
        haar_wavelet_transform(block, LL, LH, HL, HH);
        // Perform SVD on LL subband
        Mat Uc, Sc, Vtc;
        compute_svd(LL, Uc, Sc, Vtc);
        
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
    reconstruct_matrix(Uwm, extracted_watermark_S, Vtwm, extracted_watermark);
    // Normalize and convert to CV_8U
    normalize(extracted_watermark, extracted_watermark, 0, 255, NORM_MINMAX);
    extracted_watermark.convertTo(extracted_watermark, CV_8U);

    return extracted_watermark;
}

int sequential(std::chrono::milliseconds * execution_time, bool isDisplay, string original_image_path, string watermark_image_path) {
    int original_width = 512;
    int original_height = 512;
    int watermark_width = 64;
    int watermark_height = 64;
    int block_size = 4;
    int n_blocks_to_embed = 128;
    double spatial_weight = 0.33;

    string output_filename = "serial_watermarked_image";
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
        //cout << "Original image resized to "<<original_width<<"x"<<original_height << endl;
    }

    if(isDisplay) {
        // Display original image
        namedWindow("Resized Original Image", WINDOW_NORMAL);
        imshow("Resized Original Image", original_image);
        // Wait for a key press to proceed
        waitKey(0);
    }

    // Save original resized image
    bool isSaved = imwrite("resized_" + original_image_path, original_image);
    if (isSaved) {
        //cout << "Original resized image saved as '" << output_image_path << "'." << endl;
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

    Mat watermarked_image = embed_watermark(
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
        //cout << "Watermarked image saved as '" << output_image_path << "'." << endl;
    }
    else {
        cerr << "Error: Could not save watermarked image." << endl;
    }

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

    Mat extracted_watermark = extract_watermark(ext_watermarked_image, output_filename, n_blocks_to_embed, block_size, alpha);

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
        //cout << "Extracted watermark saved as '" << extracted_watermark_path << "'." << endl;
    }
    else {
        cerr << "Error: Could not save the extracted watermark." << endl;
    }
    return 0;
}
