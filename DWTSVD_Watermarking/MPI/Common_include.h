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
#include <mpi.h>

// Structure to hold block information
struct Block {
    cv::Rect location;
    double spatial_value;
    double attack_value;
    double merit;
};