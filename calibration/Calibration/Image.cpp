#include "Image.h"
#include <iostream>
#include <algorithm>
#include "Estimation.h"

using namespace cv;
using namespace std;

#define BLACK 0
#define WHITE 255
#define USED 128

#define MIN_PATH_SIZE 4

#define GRAD_THRESHOLD 10
#define MIN_LINE_LENGTH 10

#define RANSAC_LINE_ERROR 1.f
#define CORNER_CONTOUR_EPSILON 5.f

#define LONG_SIDE 9
#define SHORT_SIDE 7

//#define DEBUG

/*
  Implementation of some common image operation functions,
  like thresholding and erosion
*/

/*
	Gaussian kernel
	Returns an x by y gaussian kernel with the given sigma
	x and y have to be odd
*/
Mat GaussianKernel(const int x, const int y, const int sigma)
{
	Mat gaussKernel = Mat(y, x, CV_32F, 1.f);
	int sigmaSq = sigma * sigma;
	for (int h = 0; h < y; ++h)
	{
		for (int w = 0; w < x; ++w)
		{
			// center coordinates
			Point p(w - (x/2), h - (y/2));

			// compute gaussian value as a float
			float g = exp(-(p.x*p.x + p.y*p.y) / (2 * sigmaSq)) / (2 * 3.14159*sigmaSq);

			gaussKernel.at<float>(h, w) = g;
		}
	}
	// normalise the corners
	gaussKernel /= gaussKernel.at<float>(0, 0);
	return gaussKernel;
}

// Helper functions
bool IsInBounds(int height, int width, Point p)
{
	if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height)
	{
		return true;
	}
	return false;
}

/*
	Gaussian thresholding
	This calculates the threshold per pixel by computing the mean of the neighbourhood values
	weighted by a gaussian kernel of the same size
*/
bool GaussianThreshold(const cv::Mat& input, cv::Mat& output, int kernelSize, int constant)
{
	// Some brief error checking
	if (input.rows != output.rows || input.cols != output.cols)
	{
		return false;
	}
	if (kernelSize <= 0 || kernelSize % 2 == 0)
	{
		return false;
	}

	// construct gaussian kernel of given size
	Mat gaussKernel = GaussianKernel(kernelSize, kernelSize, kernelSize/5);
	float kernelWeight = 0;
	for (int y = 0; y < kernelSize; ++y)
	{
		for (int x = 0; x < kernelSize; ++x)
		{
			kernelWeight += gaussKernel.at<float>(y, x);
		}
	}
	/*Mat(kernelSize, kernelSize, CV_32F, 1);
	for (int i = 0; i < kernelSize; ++i) for (int j = 0; j < kernelSize; ++j) gaussKernel.at<float>(i, j) = 1;
	GaussianBlur(gaussKernel, gaussKernel, Size(kernelSize, kernelSize), 1, 1, BORDER_DEFAULT);*/

//#ifdef DEBUG
	//cout << gaussKernel << endl;
//#endif

	// centre of neighbourhood is half the kernel size plus 1:
	int centre = kernelSize / 2 + 1;
	int halfKernel = kernelSize / 2;

	// Over each pixel, for the neighbourhood around it, compute the mean
	// and threshold
	// Any points off-image are 0

	float totalMean = 0;

	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			totalMean += input.at<uint8_t>(y, x);
			continue;
			// For the neighbourhood around the pixel
			float mean = 0;
			int min = 255, max = 0;
			for (int h = -halfKernel; h <= halfKernel; ++h)
			{
				for (int w = -halfKernel; w <= halfKernel; ++w)
				{
					Point p(x+w, y+h);
					int value = 0;
					if ((p.x > 0 && p.x < input.cols) && (p.y > 0 && p.y < input.rows))
					{
						value = input.at<uint8_t>(p.y, p.x);
						if (value < min)
						{
							min = value;
						}
						if (value > max)
						{
							max = value;
						}
					}
					float weight = gaussKernel.at<float>(h + halfKernel, w + halfKernel);

					mean += weight * value;
				}
			}

			mean /= kernelWeight;// kernelSize * kernelSize; // gotta divide by total weight
			mean -= constant;
			int pixelToThreshold = input.at<uint8_t>(y, x);

			// Is the range too small?
			if (max - min < 20)
			{
				mean = 128; // if it's too small, just pick an easy bound. 
				// That way, it all gets shunted to one side to avoid salt and pepper
			}

			if (pixelToThreshold > mean)
			{
				output.at<uint8_t>(y, x) = 255;
			}
			else
			{
				output.at<uint8_t>(y, x) = 0;
			}
		}
	}

	totalMean /= input.cols*input.rows;
	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			int pixelToThreshold = input.at<uint8_t>(y, x);

			if (pixelToThreshold > 127/*totalMean*/)
			{
				output.at<uint8_t>(y, x) = 255;
			}
			else
			{
				output.at<uint8_t>(y, x) = 0;
			}
		}
	}


	return true;
}

/*
	Average Threshold
	Find the average pixel value in the image, and threshold based on that
*/
bool AverageThreshold(const cv::Mat& input, cv::Mat& output)
{
	if (input.rows != output.rows || input.cols != output.cols)
	{
		return false;
	}

	// Find average
	float average = 0;
	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			average += (float)input.at<uchar>(y,x);
		}
	}
	average /= (float)input.rows*input.cols;

	// Actually do threshold
	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			auto pixel = input.at<uchar>(y, x);
			if (pixel < average)
			{
				output.at<uchar>(y, x) = BLACK;
			}
			else
			{
				output.at<uchar>(y, x) = WHITE;
			}
		}
	}

	return true;
}

/*
	Erosion
	There are two supplied kernels for this, but I guess you can also supply your own
	The kernel is placed over every input pixel. If any pixel covered by the kernel's nonzero values is
	0, then the output pixel is 0
	Any pixel outside the image is considered 0
*/
bool Erode(const cv::Mat& input, cv::Mat& output, cv::Mat erosionKernel)
{
	// Some brief error checking
	if (input.rows != output.rows || input.cols != output.cols)
	{
		return false;
	}

	// just to make sure
	output = input.clone();

	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			int erosionSum = 0;
			for (int h = 0; h < erosionKernel.rows; ++h)
			{
				for (int w = 0; w < erosionKernel.cols; ++w)
				{
					Point p(x + w - 1, y + h - 1);
					if ((p.x > 0 && p.x < input.cols) && (p.y > 0 && p.y < input.rows))
					{
						erosionSum += input.at<uint8_t>(p.y, p.x)*erosionKernel.at<int>(h, w);
					}
				}
			}

			if (erosionSum > 0)
			{
				output.at<uint8_t>(y, x) = WHITE;
			}
			else
			{
				output.at<uint8_t>(y, x) = input.at<uint8_t>(y, x);
			}
		}
	}

	return true;
}

/*
	Find Contours
	This requires a binarised image. We search through the image for black components,
	and walk the edge of black and white until we meet with the starting point. Then, this
	whole area is given to a floodfill algorithm that replaces the black with a grey that
	signifies that this blob has been found. The image is searched further and this process
	repeats
*/
// I know this is hack but it's the best way I could get it working
const cv::Point dirs[Contour::NUM_DIRS] =
{
	Point(0,-1),
	Point(-1,-1),
	Point(-1,0),
	Point(-1,1),
	Point(0,1),
	Point(1,1),
	Point(1,0),
	Point(1,-1)
};
bool PixelIsAdjacentToWhite(const Mat& input, const Point& p)
{
	for (int i = 0; i < Contour::NUM_DIRS; ++i)
	{
		if (!IsInBounds(input.rows, input.cols, p + dirs[i]))
		{
			continue;
		}

		auto pixel = input.at<uchar>(p + dirs[i]);
		if (pixel == WHITE)
		{
			return true;
		}
	}

	return false;
}
bool FindContours(const cv::Mat& input, std::vector<Contour>& contours, bool debug)
{
	Mat img = input.clone();

#define NO_POINT Point(-1,-1)

	Point start(-1, -1);
	Contour currentContour;
	currentContour.start = NO_POINT;
	for (int y = 0; y < img.rows; ++y)
	{
		for (int x = 0; x < img.cols; ++x)
		{
			auto pixel = input.at<uint8_t>(y, x);

			// Find the beginning of a contour
			if (pixel == BLACK && PixelIsAdjacentToWhite(input, Point(x,y)))
			{
				start = Point(x, y);
				// use a separate function to walk this
				//Contour c = FindContour(img, start); // this is what's wrong
				Contour c = FloodFillEdgePixels(img, start, USED);
				if (c.path.size() > MIN_PATH_SIZE)
				{
					contours.push_back(c);
					if (debug)
					{
						imshow("a contour", img);
						waitKey(0);
					}
				}
				 
				
			}

			// We found the contour here.
			// Mark it, remove squares, then keep looking for more
		}
	}

#ifdef DEBUG
	DrawContours(img, contours);
#endif

	return true;
}

// Unit test for FindContour
void TestFindContour()
{
	// Create test mat and fill it
	Mat test1 = Mat(4, 4, CV_8U, cvScalar(0));
	test1.at<uchar>(0, 0) = WHITE;
	test1.at<uchar>(0, 1) = WHITE;
	test1.at<uchar>(0, 2) = WHITE;
	test1.at<uchar>(0, 3) = WHITE;
	test1.at<uchar>(3, 0) = WHITE;
	test1.at<uchar>(3, 1) = WHITE;
	test1.at<uchar>(3, 2) = WHITE;
	test1.at<uchar>(3, 3) = WHITE;
	test1.at<uchar>(1, 0) = WHITE;
	test1.at<uchar>(2, 0) = WHITE;
	test1.at<uchar>(1, 3) = WHITE;
	test1.at<uchar>(2, 3) = WHITE;

	cout << test1 << endl;

	//Point start(1,1);
	//Contour c = FindContour(test1, start);

	//assert(c.path.size() == 4);
	/*assert(c.path[0] == Contour::RIGHT);
	assert(c.path[1] == Contour::DOWN);
	assert(c.path[2] == Contour::LEFT);
	assert(c.path[3] == Contour::UP);*/

}

/*
	Flood fill

	Given a point in an image, flood fill 8 way on the value of the
	starting pixel. Fill with newVal
*/
Contour FloodFillEdgePixels(Mat& img, const Point& start, int newVal)
{
	Contour c;
	c.length = 0;
	c.start = start;	
	//c.path.push_back(start);

	// sanity check
	if (!IsInBounds(img.rows, img.cols, start))
	{
		return c;
	}

	auto fillVal = img.at<uchar>(start);
	vector<Point> stack;
	stack.push_back(start);

	while (!stack.empty())
	{
		// sanity
		if (stack.size() > img.cols*img.rows)
		{
			break;
		}
		
		// Get the front of the stack and remove it
		Point p = stack.back();
		stack.pop_back();

		// check if this point has been taken care of already
		if (img.at<uchar>(p) == newVal)
		{
			continue;
		}

		// If this point is an edge point, add it to the contour
		if (PixelIsAdjacentToWhite(img, p))
		{
			c.path.push_back(p);
		}

		// fill this point
		img.at<uchar>(p) = newVal;

		// push all points around it, 8-way, that are the fill value
		for (int i = -1; i <= 1; ++i)
		{
			for (int j = -1; j <= 1; ++j)
			{
				if (i == 0 && j == 0)
				{
					continue;
				}

				Point q = p + Point(i, j);
				if (IsInBounds(img.rows, img.cols, q) && img.at<uchar>(q) == fillVal)
				{
					stack.push_back(q);
				}
			}
		}
	}

	return c;
}

/*
	DEBUG
	Draw a given set of contours in an image. Should be the image they were found in
*/
void DrawContours(const cv::Mat& input, const std::vector<Contour>& contours)
{
	Mat draw = input.clone();

	// draw each contour
	for (auto& c : contours)
	{
		Point curPoint = c.start;
		for (auto& p : c.path)
		{
			draw.at<uchar>(p) = 128;
			//circle(draw, curPoint, 2, (128, 128, 128), -1);
		}
	}
	namedWindow("contours", WINDOW_NORMAL);
	imshow("contours", draw);
	waitKey(0);
}

/*
	DEBUG
	Draw a set of lines in the image as long as they go
*/
void DrawLine(const cv::Mat& input, const LineSegment l)
{
	Mat draw = input.clone();

	LineSegment topEdge{ Point(0,0), Point(input.cols-1, 0) };
	LineSegment rightEdge{ Point(input.cols-1, 0), Point(input.cols-1, input.rows-1) };
	LineSegment bottomEdge{ Point(0,input.rows-1), Point(input.cols-1, input.rows-1) };
	LineSegment leftEdge{ Point(0,0), Point(0, input.rows-1) };

	// extend line out to meet image edge
	// Get intersection of line with each image edge
	// Then find the closest two points, and those frame the line we want to draw
	auto pTop = GetIntersectionOfLines(l, topEdge);
	auto pRight = GetIntersectionOfLines(l, rightEdge);
	auto pBottom = GetIntersectionOfLines(l, bottomEdge);
	auto pLeft = GetIntersectionOfLines(l, leftEdge);

	// Some of these points will not be in the frame, and some will
	if (IsInBounds(input.rows, input.cols, pTop))
		line(draw, pTop, l.p1, (128, 128, 128), 1);
	if (IsInBounds(input.rows, input.cols, pRight))
		line(draw, pRight, l.p1, (128, 128, 128), 1);
	if (IsInBounds(input.rows, input.cols, pLeft))
		line(draw, pLeft, l.p1, (128, 128, 128), 1);
	if (IsInBounds(input.rows, input.cols, pBottom))
		line(draw, pBottom, l.p1, (128, 128, 128), 1);
#ifdef DEBUG
	namedWindow("lines", WINDOW_NORMAL);
	imshow("lines", draw);
	waitKey(0);
#endif
}

/*
	Find quadrangles in an image

	This searches through contours for "straight line sections" - 
	sections of the contour with a total, and average (over 5 or ten pixels), angular difference
	of close to 0. 
	If we find only four straight line sections that take up 90% or more of the contour then this
	is deemed a quadrangle
*/
// Helper function
int ConvolveDerivativeKernel(const int kernel[5], const int kernelDivisor, const int data[5])
{
	int result = kernel[0] * data[0] + kernel[1] * data[1] + kernel[2] * data[2] + kernel[3] * data[3] + kernel[4] * data[4];
	result /= kernelDivisor;
	result = abs(result);

	return result;
}
Point GetIntersectionOfLines(const LineSegment& l1, const LineSegment& l2)
{
	float a1 = l1.p2.y - l1.p1.y;
	float b1 = l1.p1.x - l1.p2.x;
	float c1 = -1 * l1.p1.x*a1 - l1.p1.y*b1;

	float a2 = l2.p2.y - l2.p1.y;
	float b2 = l2.p1.x - l2.p2.x;
	float c2 = -1 * l2.p1.x*a2 - l2.p1.y*b2;

	float x, y;
	// The following comes from converting both lines from 
	// the form ax + by + c = 0 (which is where I get the a1, etc)
	// to y = mx + b form, and then equating the y's and solving for x
	if ((b1 == 0 && b2 == 0) || (a1 == 0 && a2 == 0))
	{
		// Lines are both vertical or both horizontal. Return a dud point
		return Point(-1, -1);
	}
	if (b1 == 0)
	{
		x = -c1 / a1;
		y = (-a2/b2)*x + (-c2/b2);
	}
	else if (b2 == 0)
	{
		x = -c2 / a2;
		y = (-a1 / b1)*x + (-c1 / b1);
	}
	else if (a1 == 0)
	{
		y = -c1 / b1;
		x = (-b2/a2)*y + (-c2/a2);
	}
	else if (a2 == 0)
	{
		y = -c2 / b2;
		x = (-b1 / a1)*y + (-c1 / a1);
	}
	else {
		x = ((c2 / b2) - (c1 / b1)) / ((a1 / b1) - (a2 / b2));
		y = -1 * (a1 / b1)*x - c1 / b1;
	}

	

	return Point((int)x, (int)y);
}
bool CheckCornerValidity(const Contour& c, const Point& p1)
{
	bool withinBounds = false;
	for (const auto& p : c.path)
	{
		if (DistBetweenPoints(p1, p) < CORNER_CONTOUR_EPSILON)
		{
			withinBounds = true;
			break;
		}
	}

	// Corner is too far from the contour, this isn't good
	if (!withinBounds)
	{
		return false;
	}
	return true;
}
// Actual function
bool FindQuad(const Mat& img, const Contour& c, Quad& q)
{
	// get all points in a vector
	// New idea: RANSAC
	// Use RANSAC to find lines such that:
	// a line uses at least a fifth of the points in the contour
	// the error bar for inliers is really small
	// See how many lines we can get
	// If only four good lines can be found from the points,
	// This is a quadrangle! Compute the corners

	// Get all the points of the contour into a vector
	vector<Point> points;
	for (auto& p : c.path)
	{
		points.push_back(p);
	}

	// print each point fromt he contour you are currently describing?
	// Seems like RANSAC dies after getting two lines, and can't get horizontal lines...

	int minLineSize = points.size() / 5;
	vector<LineSegment> lines;
	while (true)
	{
		// Search among points for a line with RANSAC
		vector<Point> inliers;
		pair<Point, Point> seedPoints;
		inliers = FindLineInPointsRANSAC(points, minLineSize, RANSAC_LINE_ERROR, 500, seedPoints);

		if (!inliers.empty())
		{
			// remove inliers from points
			for (auto p : inliers)
			{
				int i = 0;
				for (i = 0; i < points.size(); ++i)
				{
					if (p == points[i])
					{
						break;
					}
				}
				points.erase(points.begin() + i);
			}

			// Create a line segment with the start and end of inliers
			// We define it by the two points that were used to create it
			LineSegment l;
			l.p1 = seedPoints.first;
			l.p2 = seedPoints.second;
			lines.push_back(l);
		}
		else
		{
			// No more lines can be found. Quit
			break;
		}
	}

	// Find the centroid of the contour as a heuristic for the quad centre
	// And the two furthest points for diagonal width
	Point centroid(0,0);
	float size = 0;
	for (auto& p : c.path)
	{
		centroid += p;
		for (auto& q : c.path)
		{
			float d = DistBetweenPoints(p, q);
			if (d > size)
			{
				size = d;
			}
		}
	}
	centroid.x /= c.path.size();
	centroid.y /= c.path.size();


	// Check that there are four lines
	if (lines.size() == 4)
	{
		// Compute the corners
		// These are intersections of the lines that lay within the image bounds
		// over each pair in this set of four
		// find the intersection

		int cornerIndex = 0;
		int centreX = 0, centreY = 0;

		// There's a better way to do this. 
		// Get the first intersection. Remove the first line, start with the next
		// Get the intersection between it and the next possible one. This has one possibility
		// Get the last line, intersect it with the first. 

		// Get first intersection. It's either between line 0 and line 1, or l0 and l2:
		// Try l0 and l1:
		Point corner = GetIntersectionOfLines(lines[0], lines[1]);
		LineSegment nextLine = lines[1];
		LineSegment nextOtherLine = lines[2];
		if (IsInBounds(img.rows, img.cols, corner) && DistBetweenPoints(corner, centroid) < size)
		{
			q.points[0] = corner;
			centreX += corner.x;
			centreY += corner.y;
		}
		else {
			nextLine = lines[2];
			nextOtherLine = lines[1];
			Point corner = GetIntersectionOfLines(lines[0], lines[2]);
			if (IsInBounds(img.rows, img.cols, corner) && DistBetweenPoints(corner, centroid) < size)
			{
				q.points[0] = corner;
				centreX += corner.x;
				centreY += corner.y;
			}
		}
		// Next corner is between whatever just didn't work, and the one that did
		corner = GetIntersectionOfLines(nextLine, nextOtherLine);
		LineSegment finalLine = lines[3];
		if (IsInBounds(img.rows, img.cols, corner) && DistBetweenPoints(corner, centroid) < size)
		{
			q.points[1] = corner;
			centreX += corner.x;
			centreY += corner.y;
		}
		else {
			// If this fails, nextLine and line 3
			Point corner = GetIntersectionOfLines(nextLine, lines[3]);
			finalLine = nextOtherLine;
			nextOtherLine = lines[3];
			if (IsInBounds(img.rows, img.cols, corner) && DistBetweenPoints(corner, centroid) < size)
			{
				q.points[1] = corner;
				centreX += corner.x;
				centreY += corner.y;
			}
		}
		// Next corner is between what we just connected to, and the final line. This has to be a corner
		corner = GetIntersectionOfLines(nextOtherLine, finalLine);
		if (IsInBounds(img.rows, img.cols, corner) && DistBetweenPoints(corner, centroid) < size)
		{
			q.points[2] = corner;
			centreX += corner.x;
			centreY += corner.y;
		}
		// And last but not least, final line back to line 0
		corner = GetIntersectionOfLines(finalLine, lines[0]);
		if (IsInBounds(img.rows, img.cols, corner) && DistBetweenPoints(corner, centroid) < size)
		{
			q.points[3] = corner;
			centreX += corner.x;
			centreY += corner.y;

			if (!CheckCornerValidity(c, corner))
			{
#ifdef DEBUG
				DrawLine(img, lines[0]);
				DrawLine(img, finalLine);
#endif
			}
		}

		// Better confirmation - confirm that for each corner there exists a point
		// on the contour such that each corner is within some small epsilon
		// of this point
		// This will at least catch those cases
		// The bound is pretty loose
		for (int i = 0; i < 4; ++i)
		{
			auto& c1 = q.points[i];
			if (!CheckCornerValidity(c, c1))
			{
				return false;
			}
		}

		// Compute the centre
		q.centre = Point(centreX/4, centreY/4);

		return true;
	}

	return false;
}

/*
	Given a quad, what's the length of the longest diagonal?
	This also involves figuring out which corners are diagonal. 
	For this, pick two corners, define a line, and if both other corners
	are on the same side of it, then it's not a diagonal
*/
// Helpers
float DistBetweenPoints(const Point& p1, const Point& p2)
{
	float x = (p1.x - p2.x);
	float y = (p1.y - p2.y);
	return sqrt(x*x + y*y);
}
// Actual
float GetLongestDiagonal(const Quad& q)
{
	// Just go over all pairs of points and get the longest distance
	// between two pairs
	// Since a side won't be bigger than a diagonal unless we have a really
	// weird lens
	float maxDist = 0;
	for (int i = 0; i < 4; ++i)
	{
		Point p1 = q.points[i];
		for (int j = i + 1; j < 4; ++j)
		{
			Point p2 = q.points[j];

			float dist = DistBetweenPoints(p1, p2);
			if (dist > maxDist)
			{
				maxDist = dist;
			}
		}
	}

	return maxDist;
}

/*
	Returns +ve for one side, -ve for the other. Useful for checking whether
	two points lie on the same side of the line. 
	p is the point, p1 and p2 form the line, defined as p1 to p2
*/
int PointDistToLineSigned(const Point& p, const Point& p1, const Point& p2)
{
	const int a = p2.y - p1.y;
	const int b = p1.x - p2.x;
	const int c = -1 * p1.x*a - p1.y*b;

	int d = (a*p.x + b * p.y + c) / sqrt(a*a + b * b);
	return d;
}

/*
	Comparator for quads, ordering by x coord of centre
*/
bool CompareQuadByCentreX(Quad* a, Quad* b)
{
	return a->centre.x < b->centre.x;
}

/*
	Comparator for quads, ordering by angle to centre
*/
bool CompareQuadByAngleToCentre(Quad a, Quad b)
{
	return a.angleToCentre > b.angleToCentre;
}

/*
	For two quads, does a point lie within the quadrilateral defined
	by the lines from each quad's centre, perpendicular to the quad's sides?
*/
bool DoesPointLieWithinQuadOfTwoCentres(const Point& p, const Quad& q1, const Quad& q2)
{
	// Check if the point is the same side of each line as the centre point of the line between
	// the centres
	// as that line must be within the quad 
	// have a function that takes in a line and a point and returns 1 for one side and -1 for the other

	Point centre((q1.centre.x+q2.centre.x)/2, (q1.centre.y + q2.centre.y) / 2);
	
	// construct the lines perpendicular to each side. To do this, we draw lines 
	// from the centres of the quads to the midpoints of each side. The midpoint
	// is the average of the corners
	// Note that it doesn't actually matter which midpoints we pick, as long as we pick two
	// non-opposite ones which is a pain now that I think about it
	// TODO: label quad points as diagonal or not
	// Pick corners that are adjacent. Corners 0 and 1 for one side, and 1 and 2 for the other
	Point q1SideMid1((q1.points[0].x+q1.points[1].x)/2, (q1.points[0].y + q1.points[1].y)/2); // 0 and 1
	Point q1SideMid2((q1.points[1].x + q1.points[2].x) / 2, (q1.points[1].y + q1.points[2].y) / 2); // 
	Point q2SideMid1((q2.points[0].x + q2.points[1].x) / 2, (q2.points[0].y + q2.points[1].y) / 2);
	Point q2SideMid2((q2.points[1].x + q2.points[2].x) / 2, (q2.points[1].y + q2.points[2].y) / 2);

	int value = PointDistToLineSigned(p, q1.centre, q1SideMid1)*PointDistToLineSigned(centre, q1.centre, q1SideMid1);
	if (value > 0) // they're either both positive or both negative. Thus, same side of the line
	{
		// Repeat for next lines
		value = PointDistToLineSigned(p, q1.centre, q1SideMid2)*PointDistToLineSigned(centre, q1.centre, q1SideMid2);
		if (value > 0)
		{
			value = PointDistToLineSigned(p, q2.centre, q2SideMid1)*PointDistToLineSigned(centre, q2.centre, q2SideMid1);
			if (value > 0) 
			{
				value = PointDistToLineSigned(p, q2.centre, q2SideMid2)*PointDistToLineSigned(centre, q2.centre, q2SideMid2);
				if (value > 0)
				{
					return true;
				}
			}
		}
	}

	return false;
}

/*
	Order Quads

	Find one that has only one linked corner. This has to be a corner
	Go to its linked quad. From that one, go to the next that has only two linked.
	Repeat, until you hit one.
	Save that original second quad, and repeat on its other corner.
	We now have the edge quads.
	Following this pattern, number all quads and order them
*/
void OrderQuads(const vector<Quad>& quads, vector<int>& checkerOrdering)
{
	Quad corner;
	for (const Quad& q : quads)
	{
		if (q.numLinkedCorners == 1)
		{
			corner = q;
			break;
		}
	}

	// Now search along one direction for the end
	Quad curQuad = corner;
	int numQuadsAlongSide = 1;
	do
	{
		// Find the index of the linked quad
		int newIndex = -1;
		for (int i = 0; i < 4; ++i)
		{
			if (curQuad.associatedCorners[i].first != -1)
			{
				// We alternate between quads with 4 and quads with 2
				Quad nextQuad = quads[curQuad.associatedCorners[i].second];
				if (curQuad.numLinkedCorners == 4 && nextQuad.numLinkedCorners == 2)
				{
					curQuad = nextQuad;
				}
				else if (curQuad.numLinkedCorners == 2 && nextQuad.numLinkedCorners == 4)
				{
					curQuad = nextQuad;
				}
				else if (nextQuad.numLinkedCorners == 1)
				{
					// This is the corner quad!
					curQuad = nextQuad;
				}
				newIndex = curQuad.associatedCorners[i].second;
			}
		}

		if (newIndex < 0) continue;

		// Get the next quad
		numQuadsAlongSide++;

	} while (curQuad.numLinkedCorners != 1);

	if (numQuadsAlongSide == LONG_SIDE)
	{
		
	}
	else if (numQuadsAlongSide == SHORT_SIDE)
	{

	}
	else
	{
		// Things didn't work
		checkerOrdering.clear();
		return;
	}

	// Now that we have some ordered,
}

/*
	Sort quads by x coordinate ascending
*/
bool OrderTwoQuadsByAscendingCentreX(Quad a, Quad b)
{
	return a.centre.x < b.centre.x;
}

/*
DEBUG
Draw a quad in an image. Should be the image they were found in
*/
void DrawQuadAndDisplay(const cv::Mat& input, const Quad& q)
{
	Mat draw = input.clone();

	// draw each contour
	for (auto& p : q.points)
	{
		circle(draw, p, 2, (128, 128, 128), -1);
	}
	namedWindow("quad");
	imshow("quad", draw);
	waitKey(0);
}
cv::Mat DrawQuad(const cv::Mat& input, const Quad& q)
{
	Mat draw = input.clone();

	for (auto& p : q.points)
	{
		circle(draw, q.centre, 20, (128, 128, 128), -1);
	}
	return draw;
}

void DrawQuadsNumbered(const cv::Mat& input, const std::vector<Quad>& quads)
{
	Mat draw = input.clone();
	for (Quad q : quads)
	{
		if (!IsInBounds(draw.rows, draw.cols, q.centre))
		{
			continue;
		}

		if (q.number != 0)
		{
			putText(draw, std::to_string(q.number), q.centre,
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
		}
		else {
			circle(draw, q.centre, 20, (128, 128, 128), -1);
		}

	}
	// Debug display
	imshow("Quads", draw);
	waitKey(0);
}
