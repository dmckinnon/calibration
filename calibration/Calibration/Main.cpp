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
	- Check the math with homography decomposition
	- Checker detection still very inconsistent
	- Still getting NaN sometimes after SVD. Bad homographies?
	  I keep getting so many NaNs! Why are detections non-deterministic??
	  Only line detection should be nondeterministic

	TODO:
	- Possibly do refinement on final result
	- sometimes getting sqrt of negative. Presumably from when 
	  we are numbering badly and getting a bad homography, throwing everything off.
	  We need to use only good homographies
	  Need to guarantee good checker detection
	- Read P3P



	  NEW PLAN:
	  - print separated checkerboard bigger
	  - take fixed-focus shots from small but varying angles (Rajeev - or auto focus?)
	  - remove iterating on erode, just do one iteration
	  - this should be considerably more robust
	    and should allow for better detection of errors later on
	  - Hopefully this gives a good set of solutions, and from there I can work better

	  If this continues to fail ... May. If no progress by May, then go to stereo

	  -------------------
	  Logs:
	  - some images get 33 qquads?
	  - not all get homography? Even when all quads are there?
	    This is to do with corner linking. probably a bug here
	  - Above bugs aside .... compute calibration not working. Returns all 0s



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
	TransformAndNumberQuads(I, checkerboard, Point2f(checkerboard.cols, checkerboard.rows), gtQuads);


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
	
	Mat temp = checkerboard.clone();
	// Draw all the quad centres after the transformation
	// on the image
	for (Quad q : gtQuads)
	{
		putText(temp, std::to_string(q.number), q.centre,
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
	}


	// Debug display
	imshow("debug", temp);
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

#ifdef DEBUG_DRAW_CHECKERS
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
			rectangle(temp2, q.points[0], q.centre, (128, 128, 128), CV_FILLED);
			rectangle(temp2, q.points[1], q.centre, (128, 128, 128), CV_FILLED);
			rectangle(temp2, q.points[2], q.centre, (128, 128, 128), CV_FILLED);
			rectangle(temp2, q.points[3], q.centre, (128, 128, 128), CV_FILLED);
		}
		// Debug display
		imshow("debug", temp2);
		waitKey(0);
#endif


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
#ifdef DEBUG_NUMBER_CHECKERS
		Mat temp3 = checkerboard.clone();
		for (Quad q : quads)
		{
			Vector3f x(q.centre.x, q.centre.y, 1);
			Vector3f Hx = H * x;
			Hx /= Hx(2);
			auto qCentre = Point2f(Hx(0), Hx(1));

			putText(temp3, std::to_string(q.number), qCentre,
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

			circle(temp3, qCentre, q.size, (128, 128, 128), 2);
		}
		// Debug display
		imshow("Numbered and Hd", temp3);
		waitKey(0);
#endif

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
		// TODO:
		// is K in the right coordinates?
		// How to avoid NaN?
		cout << K << endl;
		cout << K / K(2, 2) << endl;
		// print K
		// or save to a file

		for (auto& c : calibrationEstimates)
		{
			c.K = K;

			// Compute the SE3 pose too
			// We only need the first two vectors
			auto lambda = 1.f / (K.inverse() * Vector3f(c.H(0, 0), c.H(1, 0), c.H(2, 0))).norm();
			auto r1 = lambda * K.inverse() * Vector3f(c.H(0,0), c.H(1,0), c.H(2,0));
			c.r[0] = r1;// / r1.norm();
			auto r2 = lambda * K.inverse() * Vector3f(c.H(0, 1), c.H(1, 1), c.H(2, 1));
			c.r[1] = r2;// / r2.norm();
			c.r[2] = c.r[0].cross(c.r[1]);
			auto t = lambda * K.inverse() * Vector3f(c.H(0, 2), c.H(1, 2), c.H(2, 2));
			c.t = t;// / t.norm();

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

	// All the estimates should have the new parameters now

	checkerboard.release();
	return 0;
}
