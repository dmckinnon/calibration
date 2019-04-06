#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Features.h"
#include "Image.h"

#define MAX_ERODE_ITERATIONS 4 // 10

// structure for the calibration of a camera
struct Calibration
{
	Eigen::Matrix3f H;
	Eigen::Matrix3f K;
	Eigen::Vector3f r[3];
	Eigen::Matrix3f R;
	Eigen::Vector3f t;

	cv::Point2f size;

	/*
	TODO: have distortion params here
	*/

	std::vector<Quad> quads;
};

float L2norm(cv::Point a);

/* Calibration functions */
bool CheckerDetection(const cv::Mat& checkerboard, std::vector<Quad>& quads, bool debug);

bool GetHomographyAndMatchQuads(Eigen::Matrix3f& H, const cv::Mat& img, const cv::Mat& checkerboard, std::vector<Quad>& gtQuads, std::vector<Quad>& quads);

void TransformAndNumberQuads(const Eigen::Matrix3f& H, const cv::Mat& checkerboard, const cv::Point2f size, std::vector<Quad>& quads);

bool ComputeCalibration(const std::vector<Calibration>& estimates, Eigen::Matrix3f& K);