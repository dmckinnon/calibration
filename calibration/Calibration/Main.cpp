#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include "Features.h"
#include "Estimation.h"
#include "Calibration.h"
#include "Image.h"

using namespace std;
using namespace cv;
using namespace Eigen;

#define CHECKERBOARD_FILENAME "checkerboard.jpg"

//#define DEBUG

/*
	So for this next tutorial we are doing Zhang calibration. 

	Input:
	- several frames of a full view of a checkerboard
	- the checkerboard image itself

	Output: 
	- the camera matrix and distortion parameters

	Steps:
	
	Checkerboard Recognition
		Use Scarramuzza. 
		Note that this won't work on images with high distortion as we rely on straight lines
		for high distortion you need to just do corner detection, but soecifically tuned for saddles
		This also may not work so well on captured images. We shall see

	Initial estimation
		Well, we have matches - get a homography. We then decompose this homography into the camera matrix and the extrinsic pose
		of the camera from the checkerboard

	Refinement
		All this is done over each frame. Once we have all these initial estimations, we perform bundle adjustment over all the parameters
		to refine the camera matrix

	TO add:
	- distortion model


	TODO:
	- Algorithm to number quads. This then is applied after the homography
	- Then 

	LOG:
	- Now we can find the corners (to debug)
*/
int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "Missing command line arguments!" << endl;
		cout << "Format: calibration.exe <FolderToImages> numImages" << endl;
		cout << "Images are expected to be named 1.jpg, 2.jpg, etc ... " << endl;
		cout << "The ground truth checkerboard pattern is expected here and named " << CHECKERBOARD_FILENAME << endl;
		exit(1);
	}

	// Take in the folder and the number of images
	string folder = argv[1];
	int numImages = stoi(argv[2]);

#ifdef DEBUG
	std::string debugWindowName = "debug image";
	namedWindow(debugWindowName);
#endif

	/*******************************************/
	/* Get data from ground truth checkerboard */

	// Get ground truth checkerboard image
	Mat checkerboard = imread(folder + "\\" + CHECKERBOARD_FILENAME, IMREAD_GRAYSCALE);
	
		// Display for debug
#ifdef DEBUG
	Mat temp = checkerboard.clone();
	// Debug display
	imshow(debugWindowName, temp);
	waitKey(0);
#endif

	vector<Quad> gtQuads;
	if (!CheckerDetection(checkerboard, gtQuads))
	{
		return 1;
	}
	checkerboard.release();

	/*********************************/
	/* Get data from captured images */

	// For each image in the folder, 
	// get an estimate of the calibration parameters and store these in a vector of calibration structures
	// Each of these has a possible camera matrix and extrinsics
	// By the end all these camera matrices should be the same
	vector<Calibration> calibrationEstimates;
	for (int image = 0; image < numImages; ++image)
	{
		// Read in the image
		Mat img = imread(folder + to_string(image+1) + ".jpg", IMREAD_GRAYSCALE);
	
		// Get the quads in the image
		vector<Quad> quads;
		if (!CheckerDetection(img, gtQuads))
		{
			cout << "Bad image for checkers in image " << image + 1 << endl;
			continue;
		}
		if (quads.empty())
		{
			cout << "No quads in image " << image + 1 << endl;
			continue;
		}

		// set up matches and create homography
		vector<pair<Point, Point>> matches = MatchCornersForHomography(gtQuads, quads);
		//assert(checkerboardFeatures.size() == features.size());
		for (unsigned int i = 0; i < quads.size(); ++i)
		{
			// Find the quads that have one corner only
			// if there isn't four ... may just quit
			// figure out which is the long side and short side by relative distance
			// This should be a reasonable heuristic
			// Then associate them with the gt points, create pairs that match
			//matches.push_back(make_pair(checkerboardFeatures[i].p, features[i].p));
		}
		
		Matrix3f H;
		if (!GetHomographyFromMatches(matches, H))
		{
			cout << "Failed to find homography for image " << image + 1 << endl;
			continue;
		}

		// Associate all corners from these quads for the purposes of optimisation
		AssociateAllCorners(gtQuads, quads);

		// Decompose into K matrix and extrinsics
		Matrix3f K, T;
		if (!ComputeIntrinsicsAndExtrinsicFromHomography(H, K, T))
		{
			cout << "Failed to compute intrinsics for image " << image + 1 << endl;
		}

		

		// Store
		Calibration c;
		c.K = K;
		c.R << T(0,0), T(0,1), T(0,2),
			   T(1,0), T(1,1), T(1,2),
			     0   ,    0  ,   0;
		c.t << T(2, 0), T(2, 1), T(2, 2);
		calibrationEstimates.push_back(c);

		// free the memory
		img.release();
	}

	// We need a minimum number of estimates for this to work
	if (calibrationEstimates.size() < 3)
	{
		cout << "Not enough images worked for calibration to be viable" << endl;
		return 1;
	}

	/********************/
	/* Refine estimates */

	return 0;
}