#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Features.h"
#include "Image.h"

#define MAX_ERODE_ITERATIONS 4

// structure for the calibration of a camera
struct Calibration
{
	Eigen::Matrix3f K;
	Eigen::Matrix3f R;
	Eigen::Vector3f t;

	// And any other camera parameters
};

/* Calibration functions */
bool EnumerateCheckerboardCorners(std::vector<Feature>& features, cv::Point& size, const bool gt);

bool CheckerDetection(const cv::Mat& checkerboard, std::vector<Quad>& quads);

bool ComputeIntrinsicsAndExtrinsicFromHomography(const Eigen::Matrix3f& H, Eigen::Matrix3f& K, Eigen::Matrix3f& T);

std::vector<std::pair<cv::Point, cv::Point>> MatchCornersForHomography(std::vector<Quad>& gtQuads, std::vector<Quad>& quads)

void AssociateAllCorners(std::vector<Quad>& gtQuads, std::vector<Quad>& quads);