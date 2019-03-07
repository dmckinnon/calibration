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

#define DEBUG

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

	Test this all with P3P!


	Issues:
	- Cannot get a good homography
	- With a mildly inaccurate homography ... how do we number them correctly?
	- Numbering - need to record indices
	- Homography solved. Now fix numbering

	TODO:
	- refinement
	- Read P3P

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
#ifdef DEBUG
	Mat temp = checkerboard.clone();
#endif

	vector<Quad> gtQuads;
	cout << "Finding checkers in synthetic image" << endl;
	if (!CheckerDetection(checkerboard, gtQuads, false))
	{
		return 1;
	}
	//checkerboard.release();

	// identity homography for gt quads to not transform them
	cout << "Numbering synthetic checkers" << endl;
	Matrix3f I;
	I << 1, 0, 0,
		 0, 1, 0,
		 0, 0, 1;
	// initial numbering
	int topleft = 0;
	int topright = 0;
	int bottomleft = 0;
	int bottomright = 0;
	for (int i = 0; i < gtQuads.size(); ++i)
	{
		const Quad& q = gtQuads[i];
		// topleft
		if ((float)q.centre.x < gtQuads[topleft].centre.x*0.9f || (float)q.centre.y < gtQuads[topleft].centre.y*0.9f)
		{
			topleft = i;
		}
		// topright
		if ((float)q.centre.x > gtQuads[topright].centre.x*1.1f || (float)q.centre.y < gtQuads[topright].centre.y*0.9f)
		{
			topright = i;
		}
		// bottom left
		if ((float)q.centre.x < gtQuads[bottomleft].centre.x*0.9f || (float)q.centre.y > gtQuads[bottomleft].centre.y*1.1f)
		{
			bottomleft = i;
		}
		// bottom right
		if ((float)q.centre.x > gtQuads[bottomright].centre.x*1.1f || (float)q.centre.y > gtQuads[bottomright].centre.y*1.1f)
		{
			bottomright = i;
		}
	}
	gtQuads[topleft].number = 1;
	gtQuads[topright].number = 5;
	gtQuads[bottomleft].number = 28;
	gtQuads[bottomright].number = 32;
	//TransformAndNumberQuads(I, gtQuads);


	// DEBUG
	// COnfirm that all the quads are good
	for (Quad q : gtQuads)
	{
		cout << "Quad number " << q.number << " has centre " << q.centre << endl;
		for (int i = 0; i < 4; ++i)
		{
			Quad q2;
			bool found = false;
			for (Quad& qt : gtQuads)
			{
				if (qt.id == q.associatedCorners[i].first)
				{
					q2 = qt;
					found = true;
					break;
				}
			}
			if (!found) continue;
			cout << "\tcorner " << q.points[i] << " connects to quad " << q2.number << " at corner " << q.associatedCorners[i].second << endl;
		}
	}

#ifdef DEBUG
	

	// Draw all the quad centres after the transformation
	// on the image
	for (Quad q : gtQuads)
	{
		putText(temp, std::to_string(q.number), q.centre,
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
	}


	// Debug display
	imshow(debugWindowName, temp);
	waitKey(0);
#endif



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
		Mat img = imread(folder + "\\" + to_string(image+1) + ".jpg", IMREAD_GRAYSCALE);
		cout << "Reading image " << folder + "\\" + to_string(image + 1) + ".jpg" << endl;
	
		// Get the quads in the image
		vector<Quad> quads;
		cout << "Finding checkers in captured image" << endl;
		if (!CheckerDetection(img, quads, false))
		{
			cout << "Bad image for checkers in image " << image + 1 << endl;
			continue;
		}
		if (quads.empty())
		{
			cout << "No quads in image " << image + 1 << endl;
			continue;
		}
		cout << "Found " << quads.size() << " quads" << endl;

		// DEBUG
		// COnfirm that all the quads are good
		Mat temp2 = img.clone();
		for (Quad q : quads)
		{
			cout << "Quad id " << q.id << " has centre " << q.centre << endl;
			for (int i = 0; i < 4; ++i)
			{
				Quad q2;
				bool found = false;
				for (Quad& qt : gtQuads)
				{
					if (qt.id == q.associatedCorners[i].first)
					{
						q2 = qt;
						found = true;
						break;
					}
				}
				if (!found) continue;
				cout << "\tcorner " << q.points[i] << " connects to quad " << q2.number << " at corner " << q.associatedCorners[i].second << endl;
			}

			putText(temp2, std::to_string(q.id), q.centre,
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(128, 128, 128), 1, CV_AA);
		}
		// Debug display
		imshow("debug", temp2);
		waitKey(0);


	

		// set up matches and create homography
		cout << "Finding homography for captured checkers" << endl;
		Matrix3f H;
		if (!GetHomographyAndMatchQuads(H, img, checkerboard, gtQuads, quads))
		{
			cout << "Failed to find homography for image " << image + 1 << endl;
			continue;
		}

		// DEBUG
		// Draw the homographied quads
		Mat temp3 = checkerboard.clone();
		for (Quad q : quads)
		{
			putText(temp3, std::to_string(q.number), q.centre,
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

			circle(temp3, q.centre, 20, (128, 128, 128), 2);
		}
		// Debug display
		imshow("Numbered and Hd", temp3);
		waitKey(0);

		// Decompose into K matrix and extrinsics
		Matrix3f K, T;
		cout << "Computing extrinsics and intrinsics" << endl;
		if (!ComputeIntrinsicsAndExtrinsicFromHomography(H, K, T))
		{
			cout << "Failed to compute intrinsics for image " << image + 1 << endl;
			continue;
		}  

		// number for debug

		// Store
		Calibration c;
		c.quads = quads;
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