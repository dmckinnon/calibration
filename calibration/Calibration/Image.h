#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>

struct Contour
{
	static enum DIRECTION
	{
		UP, 
		UPLEFT, 
		LEFT,
		DOWNLEFT,
		DOWN,
		DOWNRIGHT,
		RIGHT,
		UPRIGHT,
		NUM_DIRS
	};
	int length;
	//std::vector<DIRECTION> path;
	std::vector<cv::Point> path;
	cv::Point start;
};

struct Quad
{
	cv::Point points[4];
	cv::Point2f centre;

	int id;
	int number;
	std::pair<int, int> associatedCorners[4];
	int numLinkedCorners;
	float size;
	float angleToCentre;
};

struct LineSegment
{
	cv::Point p1;
	cv::Point p2;
};

/*
	Prototypes of some common image operation functions like
	thresholding and erosion
*/

const cv::Mat cross = (cv::Mat_<int>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
const cv::Mat rect = (cv::Mat_<int>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);

// Requires that kernelSize be odd and positive
bool GaussianThreshold(const cv::Mat& input, cv::Mat& output, int kernelSize, int constant);

bool IsInBounds(int height, int width, cv::Point p);

// Erosion using one of the supplied kernels
bool Erode(const cv::Mat& input, cv::Mat& output, cv::Mat erosionKernel);

// Find all contours in a binarised image
bool FindContours(const cv::Mat& input, std::vector<Contour>& contours, bool debug=false);

// DEBUG - draw all contours in an image
void DrawContours(const cv::Mat& input, const std::vector<Contour>& contours);

// Find a single contour given a starting point
//Contour FindContour(const cv::Mat& input, const cv::Point& start);
void TestFindContour();

// Floodfill part of an image, given a starting point, using its value
// Changes all of that value, that touch it, to the second value
Contour FloodFillEdgePixels(cv::Mat& img, const cv::Point& start, int newVal);

// Find a quadrangle in a contour, or return false if it isn't confident
bool FindQuad(const cv::Mat& img, const Contour& c, Quad& q);

// Distance between two points
float DistBetweenPoints(const cv::Point& p1, const cv::Point& p2);

// Distance from point to line, signed
int PointDistToLineSigned(const cv::Point& p, const cv::Point& p1, const cv::Point& p2);

// Compare quads
bool CompareQuadByCentreX(Quad a, Quad b);
bool CompareQuadByAngleToCentre(Quad a, Quad b);

//  Intersection of two lines
cv::Point GetIntersectionOfLines(const LineSegment& l1, const LineSegment& l2);

// Does a point lie within the quad defined by two other points?
bool DoesPointLieWithinQuadOfTwoCentres(const cv::Point& p, const Quad& q1, const Quad& q2);

// Find the length of the longest diagonal of a quad
float GetLongestDiagonal(const Quad& q);

bool OrderTwoQuadsByAscendingCentreX(Quad a, Quad b);

// DEBUG - draw quad in an image
void DrawQuad(const cv::Mat& input, const Quad& q);
