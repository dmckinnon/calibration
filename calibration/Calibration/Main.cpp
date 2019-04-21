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
//#define DEBUG_DRAW_CHECKERS
//#define DEBUG_NUMBER_CHECKERS
//#define DEBUG_CALIBRATION

/*
	This tutorial is Zhang calibration. See README for details

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


	  -------------------
	  Issues:
	  - some images get 33 qquads?
	  - not all get homography? Even when all quads are there?
	    This is to do with corner linking. probably a bug here

	  - Refinement jacobians are wrong
	  - Initial estimate is wrong
	  - corner linking is still buggy

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

	/*******************************************/
	/* Get data from ground truth checkerboard */

	// Get ground truth checkerboard image
	Mat checkerboard = imread(folder + "\\" + CHECKERBOARD_FILENAME, IMREAD_GRAYSCALE);

	/*
		Realistically, we should have a file that just details the precise points
		in this image, as it should be computer-generated. But it was good test data so I just detect them. 
	*/
	vector<Quad> gtQuads;
	cout << "Finding checkers in synthetic image" << endl;
	if (!CheckerDetection(checkerboard, gtQuads, false))
	{
		checkerboard.release();
		cout << "Could not detect checkers in synthetic image" << endl;
		return 1;
	}

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
	TransformAndNumberQuads(I, checkerboard, Point2f(checkerboard.cols, checkerboard.rows), gtQuads);


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
		int its = 0;
		bool skip = false;
		while (!CheckerDetection(img, quads, false))
		{
			cout << "Bad image for checkers in image " << image + 1 << endl;
			its++;
			quads.clear();
			if (its >= 5)
			{
				skip = true;
				break;
			}
			
		}
		if (quads.empty() || skip)
		{
			cout << "No quads in image " << image + 1 << endl;
			continue;
		}
		cout << "Found " << quads.size() << " quads" << endl;

		// set up matches and create homography
		cout << "Finding homography for captured checkers" << endl;
		Matrix3f H;
		if (!GetHomographyAndMatchQuads(H, img, checkerboard, gtQuads, quads))
		{
			cout << "Failed to find homography for image " << image + 1 << endl;
			continue;
		}

		// Should there be homography refinement here?
		// Yes. Yes there should be. I just haven't added it yet

		// Store
		// Need to store all our homographies in non-normalised coords
		// Multiply on the right by the normalisation
		Calibration c;
		c.H = H.inverse(); 
		c.H /= c.H(2, 2);
		c.quads = quads;
		c.size = Point2f(img.cols, img.rows);
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

	/************************/
	/* Compute calibration */
	Matrix3f K;
	if (ComputeCalibration(calibrationEstimates, K))
	{
		cout << "Initial K: " << endl << K << endl;

		for (auto& c : calibrationEstimates)
		{
			c.K = K;

			// Compute the SE3 pose too
			// We only need the first two vectors
			auto lambda = 1.f / (K.inverse() * Vector3f(c.H(0, 0), c.H(1, 0), c.H(2, 0))).norm();
			auto r1 = lambda * K.inverse() * Vector3f(c.H(0,0), c.H(1,0), c.H(2,0));
			c.r[0] = r1;
			auto r2 = lambda * K.inverse() * Vector3f(c.H(0, 1), c.H(1, 1), c.H(2, 1));
			c.r[1] = r2;
			c.r[2] = c.r[0].cross(c.r[1]);
			auto t = lambda * K.inverse() * Vector3f(c.H(0, 2), c.H(1, 2), c.H(2, 2));
			c.t = t;

			c.R << c.r[0][0], c.r[1][0], c.r[2][0],
				c.r[0][1], c.r[1][1], c.r[2][1],
				c.r[0][2], c.r[1][2], c.r[2][2];

			// TODO:
			// Compute a proper rotation using Zhang Appendix C
			// The following computation comes from Zhang, Appendix C
			
			// We need to compute the singular value decomposition of Q
			// where Q is the approximation to the true R. That is to say, Q = c.R 
			// at this point
			// Let SVD(Q) = USV^T
			// Then R = UV^T is the best rotation matrix that approximates Q
			// We then put this into c.R
			BDCSVD<MatrixXf> svd(c.R, ComputeFullU | ComputeFullV);
			if (!svd.computeV())
				return -1;
			auto& V = svd.matrixV();
			if (!svd.computeU())
				return -1;
			auto& U = svd.matrixU();

			c.R = U * V.transpose();


			// Now test
			// (H*N).inverse() = K [r1 r2 t]
			// so ...
			// H = HN takes image coords of captured to the normalised gt plane
			// so HN.inverse() is estimated
#ifdef DEBUG_CALIBRATION
			Mat temp4 = checkerboard.clone();
			for (Quad q : c.quads)
			{
				Matrix3f H;
				H << c.r[0][0], c.r[1][0], c.t[0],
					 c.r[0][1], c.r[1][1], c.t[1],
					 c.r[0][2], c.r[1][2], c.t[2];
				H = K * H;
				H = H.inverse().eval();
				H /= H(2, 2);
				Vector3f x(q.centre.x, q.centre.y, 1);
				Vector3f Hx = H*x;
				Hx /= Hx(2);
				auto qCentre = Point2f(Hx(0), Hx(1));

				putText(temp4, std::to_string(q.number), qCentre,
					FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

				circle(temp4, qCentre, 20, (128, 128, 128), 2);
			}
			// Debug display
			imshow("With estimated cal and pose", temp4);
			waitKey(0);
#endif
		}
	} 
	else
	{
		cout << "Failed to compute calibration" << endl;
		return -1;
	}

	// For ease of computation in refinement, create a map from number to quad for gtQuads
	map<int, Quad> gtQuadMap;
	for (Quad& q : gtQuads)
	{
		gtQuadMap[q.number] = q;
	}

	// We have an initial estimate. Now do refinement on this
	if (!RefineCalibration(calibrationEstimates, gtQuadMap))
	{
		cout << "Failed to refine our calibration" << endl;
		return false;
	}

	
	// TODO - have a different measure in y
	// or print them so they are the same
	//calibrationEstimates[0].K *= mmPerPixelInX;
	//K(2, 2) = 1;

	// All the estimates should have the new parameters now
	cout << "K: " << endl << calibrationEstimates[0].K << endl;

	checkerboard.release();
	return 0;
}
/*
1774.53 21.1188 848.329
0    1702 40.9809
0       0       1
*/