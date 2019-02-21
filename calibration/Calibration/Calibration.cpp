#include "Calibration.h"
#include "Estimation.h"
#include "Image.h"
#include <iostream>
#include <algorithm>
#include <iterator>

using namespace cv;
using namespace std;
using namespace Eigen;

#define RECT true
#define CROSS false

//#define DEBUG

/*
	Align checkerboard.

	This takes the features detected on a checkerboard, and enumerates them such that the
	corners are enumerated thusly:

	1 2 3 ... X
	...
	X+1 ..... N

	for N corners, in a rectangular pattern. This is how we have enumerated the features
	on the ground truth checkerboard, so now we can align the two and construct a homography
	between them. 

	Scarramuzza
*/
// Support functions
float L2norm(Point a)
{
	return sqrt(a.x*a.x + a.y*a.y);
}
// Actual Function
bool CheckerDetection(const Mat& checkerboard, vector<Quad>& quads, bool debug)
{
	// Make a copy
	Mat img = checkerboard.clone();

	// Threshold the image
	// Using OpenCV's example, gonna use a kernelSize of 11 and a constant of 2
	// pick kernel size based on image size
	Mat temp = img.clone();
	if (!GaussianThreshold(temp, img, 11, 2))
	{
		return false;
	}

#ifdef DEBUG
	namedWindow("threshold", WINDOW_NORMAL);
	imshow("threshold", img);
	if (debug)
		waitKey(0);
#endif

	// Now we iterate over eroding and checking for quadrangles
	// Erode alternating with the rect kernel and the cross kernel
	// We stop when the iteration has produced no more quads than the previous iteration
	// erode
	// search for blobs
	// match quads from blobs
	// find quads in previous run
	// Finally, we link the quadrangles
	
	bool kernelCrossOrRect = RECT;
	int quadID = 0;
	for (int its = 0; its < MAX_ERODE_ITERATIONS; ++its)
	{
		// erode the image
		Mat erode = img.clone();
		auto kernel = kernelCrossOrRect? rect : cross;
		kernelCrossOrRect = !kernelCrossOrRect;
		if (!Erode(img, erode, kernel))
		{
			continue;
		}
		img = erode;

#ifdef DEBUG
		namedWindow("erode", WINDOW_NORMAL);
		imshow("erode", erode);
		if (debug) waitKey(0);
#endif

		// Find contours
		vector<Contour> contours;
		if (!FindContours(img, contours/*, debug*/))
		{
			continue;
		}

#ifdef DEBUG
		// draw all the contours onto the eroded image
		if (debug) DrawContours(img, contours);
#endif

		// get quadrangles from contours
		vector<Quad> quadsThisIteration;
		for (auto& c : contours)
		{
			Quad q;
			if (FindQuad(img, c, q))
			{
				q.id = quadID++;
				// Fill with obviously bad IDs
				q.associatedCorners[0] = pair<int, int>(-1, -1);
				q.associatedCorners[1] = pair<int, int>(-1, -1);
				q.associatedCorners[2] = pair<int, int>(-1, -1);
				q.associatedCorners[3] = pair<int, int>(-1, -1);
				q.numLinkedCorners = 0;
				quadsThisIteration.push_back(q);
#ifdef DEBUG
				// draw all the quads onto the image
				if (debug) DrawQuad(img, q);
#endif
			}
		}

		// Match with previous set of quads
		// For each quad found this iteration,
		// Is there one with a centre close enough (meaning within a quarter the longest diagonal)
		// to this one, in the bigger pool of quads? If so, keep the existing and move on
		// If not, add this one to the pool
		if (!quads.empty())
		{
			for (const Quad& q1 : quadsThisIteration)
			{
				bool found = false;
				Quad q;
				for (const Quad& q2 : quads)
				{
					float diagLength = GetLongestDiagonal(q2) / 4.f;
					if (DistBetweenPoints(q1.centre, q2.centre) < diagLength)
					{
						Mat newErode = erode.clone();
						Mat orig = img.clone();
						if (debug)
						{
							DrawQuad(orig, q2);
							DrawQuad(newErode, q1);
						}

						// This quad already exists. No need to search further
						found = true;
						break;
					}
				}
				
				if (!found)
				{
					// This quad wasn't found in previous iterations. Add it
					quads.push_back(q1);
					Mat newErode = erode.clone();
					if (debug) DrawQuad(newErode, q1);
				}
			}
		}
		else {
			// No quads exist yet. Keep everything
			for (Quad& q : quadsThisIteration)
			{
				quads.push_back(q);
			}
		}

#ifdef DEBUG
		//destroyAllWindows();
#endif
	}

	// Link corners
	// For each pair of quads, find any corners they share
	// Note these links in an array, where the index of the quad's own corner
	// holds a pair of the ID of the other quad, plus the corner index it links to
	for (int i = 0; i < quads.size(); ++i)
	{
		Quad& q1 = quads[i];
		const float diag1 = GetLongestDiagonal(q1);
		for (int j = i; j < quads.size(); ++j)
		{
			Quad& q2 = quads[j];

			// Sanity check - if their centres are further away than twice the longest diagonal of the first quad, 
			// ignore this quad
			if (DistBetweenPoints(q1.centre, q2.centre) > 1.5 * diag1)
			{
				continue;
			}

			// For each corner of q1, does it lie within the rectangle created
			// by the two centres:
			/*
			|------c2
			|	    |
			|       |
			c1------|
			*/
			Point corner1;
			int index1 = -1;
			bool validCorner = false;
			for (int k = 0; k < 4; ++k)
			{
				if (DoesPointLieWithinQuadOfTwoCentres(q1.points[k], q1, q2))
				{
					validCorner = true;
					corner1 = q1.points[k];
					index1 = k;
					break;
				}
			}

			if (!validCorner)
			{
				continue;
			}

			// Now repeat for the corners of q2:
			Point corner2;
			int index2 = -1;
			validCorner = false;
			for (int k = 0; k < 4; ++k)
			{
				if (DoesPointLieWithinQuadOfTwoCentres(q2.points[k], q1, q2))
				{
					validCorner = true;
					corner2 = q2.points[k];
					index2 = k;
					break;
				}
			}

			if (!validCorner)
			{
				continue;
			}

			// Ok we have two corners that are the same corner
			// Associate these, and then set the actual point
			// to the average of the corners, as that is where the true corner is
			Point corner((corner1.x+corner2.x)/2, (corner1.y+corner2.y)/2);
			q1.points[index1] = corner;
			q2.points[index2] = corner;
			q1.associatedCorners[index1] = pair<int, int>(q2.id, index2);
			q2.associatedCorners[index2] = pair<int, int>(q1.id, index1);
			q1.numLinkedCorners++;
			q2.numLinkedCorners++;
		}
	}

#ifdef DEBUG_CORNERS
	
	// for each quad, draw all the matching corners
	for (int i = 0; i < quads.size(); ++i)
	{
		Quad q1 = quads[i];

		Mat cornerImg = checkerboard.clone();

		for (int j = 0; j < 4; ++j)
		{
			if (q1.associatedCorners[j].first != -1)
			{
				Quad q2;
				bool found = false;
				for (Quad& q : quads)
				{
					if (q.id == q1.associatedCorners[j].first)
					{
						q2 = q;
						found = true; 
						break;
					}
				}
				if (!found) continue;
				rectangle(cornerImg, q1.centre, q2.centre/*points[q1.associatedCorners[j].second]*/, Scalar(128, 128, 128), CV_FILLED);
				line(cornerImg, q1.points[0], q1.points[1], Scalar(128, 128, 128), 2);
				line(cornerImg, q1.points[1], q1.points[2], Scalar(128, 128, 128), 2);
				line(cornerImg, q1.points[2], q1.points[3], Scalar(128, 128, 128), 2);
				line(cornerImg, q1.points[3], q1.points[0], Scalar(128, 128, 128), 2);
				line(cornerImg, q2.points[0], q2.points[1], Scalar(128, 128, 128), 2);
				line(cornerImg, q2.points[1], q2.points[2], Scalar(128, 128, 128), 2);
				line(cornerImg, q2.points[2], q2.points[3], Scalar(128, 128, 128), 2);
				line(cornerImg, q2.points[3], q2.points[0], Scalar(128, 128, 128), 2);
				break;
			}
		}

		

		imshow("cornerAssociation", cornerImg);
		waitKey(0);
	}

	
#endif

	// Make sure at least 90% of the desired number of quads have been found
	if (quads.size() < 24)
	{
		return false;

	}

	return true;
}

/*
Find a corner quad from an edge quad, given a root edge quad and a branch to go down
*/
int FindCornerFromEdgeQuad(const Quad& root, const Quad& branch, vector<Quad>& quads, Quad& corner)
{
	Quad curQuad = branch;
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
				// Check just to make sure we aren't going backwards - skip the root
				if (nextQuad.centre == root.centre)
				{
					continue;
				}


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
					corner = curQuad;
					break;
				}
				newIndex = curQuad.associatedCorners[i].second;
			}
		}

		if (newIndex < 0) continue;

		// Get the next quad
		numQuadsAlongSide++;

	} while (curQuad.numLinkedCorners != 1);

	return numQuadsAlongSide;
}

/*
	Match the four extreme corners for the purposes of a homography
*/
// Helper
float GetReprojectionError(const vector<Quad>& gtQuads, const vector<Quad>& quads, const Matrix3f& H)
{

	// Find the closest quad in gt set and get error
	float e = 0;
	for (Quad q : quads)
	{
		Vector3f x(q.centre.x, q.centre.y, 1);
		Vector3f Hx = H * x;
		Hx / Hx(2);
		q.centre = Point(Hx(0), Hx(1));

		Quad closest;
		float minDist = 100000;
		for (Quad q2 : gtQuads)
		{
			auto d = DistBetweenPoints(q2.centre, q.centre);
			if (d < minDist)
			{
				minDist = d;
				closest = q2;
			}
		}

		e += L2norm(closest.centre - q.centre);
	}

	return e;
}
// Actual Function
bool GetHomographyAndMatchQuads(Matrix3f& H, const Mat& img, vector<Quad>& gtQuads, vector<Quad>& quads)
{
	vector<pair<Point, Point>> matches;

	// Find the four corners of the gt quads and mark them
	// The topmost leftmost is 1, rightmost is 5, bottom left 28, bottom right 32
	Quad topleft = gtQuads[0];
	Quad topright = gtQuads[0];
	Quad bottomleft = gtQuads[0];
	Quad bottomright = gtQuads[0];
	for (const Quad& q : gtQuads)
	{
		// topleft
		if ((float)q.centre.x < topleft.centre.x*0.9f || (float)q.centre.y < topleft.centre.y*0.9f)
		{
			topleft = q;
		}
		// topright
		if ((float)q.centre.x > topright.centre.x*1.1f || (float)q.centre.y < topright.centre.y*0.9f)
		{
			topright = q;
		}
		// bottom left
		if ((float)q.centre.x < bottomleft.centre.x*0.9f || (float)q.centre.y > bottomleft.centre.y*1.1f)
		{
			bottomleft = q;
		}
		// bottom right
		if ((float)q.centre.x > bottomright.centre.x*1.1f || (float)q.centre.y > bottomright.centre.y*1.1f)
		{
			bottomright = q;
		}
	}

	// reiterate, just in case it was overwritten
	topleft.number = 1;
	topright.number = 5;
	bottomleft.number = 28;
	bottomright.number = 32;

	// Find the corners of the detected quads
	// Find a corner on the left side of the image
	// Find both connecting corners
	// Number these, then number the final corner

	// Find the corners
	Quad corners[4];
	int index = 0;
	for (const Quad& q : quads)
	{
		if (q.numLinkedCorners == 1)
		{
			corners[index] = q;
			index ++;
			if (index == 4) break;
		}
	}

	Mat temp2 = img.clone();
	for (int i = 0; i < 4; ++i)
	{
		

		circle(temp2, corners[i].centre, 20, (128, 128, 128), -1);
	}
	// Debug display
	imshow("corners", temp2);
	waitKey(0);

	// New, easier idea:
	// Try every combination of fitting the gt corners to the captured corners
	// if we have fewer than four, then either only use two corner checkers, but use four of their points
	// or find a fourth checker by using a connecting one, and dsearching for a captured checker with 3 connections
	//
	// Pick a combination - create a homography. 
	// If homography, test it
	// warp all corners, and find the reprojection error
	// Pick the H with the smallest reprojection error

	if (index == 4)
	{
		// We have all four corners.
		// Test all 24 possibilities for minimal reprojection error
		int indices[] = { 0,1,2,3 };

		// Iterate over all permutations
		float minError = 100000000;
		int perm[] = { 0,1,2,3 };
		Matrix3f homography;
		do {
			// the gt corners 0,1,2,3 are associated with indices[0],[1],[2],[3] of captured corners
			vector<pair<Point, Point>> matches;
			matches.push_back(pair<Point, Point>(corners[indices[0]].centre, topleft.centre));
			matches.push_back(pair<Point, Point>(corners[indices[1]].centre, topright.centre));
			matches.push_back(pair<Point, Point>(corners[indices[2]].centre, bottomleft.centre));
			matches.push_back(pair<Point, Point>(corners[indices[3]].centre, bottomright.centre));

			Matrix3f h;
			if (!GetHomographyFromMatches(matches, h))
			{
				// This permutation wasn't good enough
				continue;
			}

			// DEBUG
			// Draw the homographied quads
			Mat temp3 = img.clone();
			for (Quad q : quads)
			{
				Vector3f size(L2norm(q.points[0] - q.points[1]), 0, 1);
				Vector3f Hsize = h * size;
				Hsize /= Hsize(2);
				float s = Hsize(0);
				s = s < 0 ? 2 : s;

				Vector3f x(q.centre.x, q.centre.y, 1);
				Vector3f Hx = h * x;
				Hx / Hx(2);
				auto centre = Point(Hx(0), Hx(1));



				circle(temp3, centre, s / 2, (128, 128, 128), -1);
			} 
			// Debug display
			imshow("permutation", temp3);
			waitKey(0);


			// Transform all captured quads with H

			float e = GetReprojectionError(gtQuads, quads, h);
  			if (e < minError)
			{
				minError = e;
				perm[0] = indices[0];
				perm[1] = indices[1];
				perm[2] = indices[2];
				perm[3] = indices[3];
				homography = h;
			}

		} while (std::next_permutation(indices, indices + 4));

		if (minError == 100000000)
		{
			// Nothing worked. Just bail
			return false;
		}

		cout << "The minimum error was " << minError << endl;

		H = homography;
	}
	else if (index == 3 || index == 2)
	{
		// Just use two. It's simplest
		// Or, for now, could just throw it away?

		// For simplicity, throw it away
		// we'll see how well we do on other images
		return false;
	}
	else
	{
		// This image was clearly pretty bad. Throw it away
		return false;
	}

	// Now number quads
	TransformAndNumberQuads(H, quads);

	return true;
}

/*
	Transform and number quads

	H is expected to transform the plane the quads are in to the ground truth plane
	where is it a trivial matter to scan through and assign the correct number to each quad

	We use a bound of half the transformed top left quad's diagonal as the measure for whether
	or not another quad is in the same column or row

*/
void TransformAndNumberQuads(const Eigen::Matrix3f& H, std::vector<Quad>& quads)
{
	// First, transform all quads
	for (Quad& q : quads)
	{
		Vector3f x(q.centre.x, q.centre.y, 1);
		Vector3f Hx = H * x;
		Hx / Hx(2);
		q.centre = Point(Hx(0), Hx(1));
	}

	// Second, copy vector locally for modification
	vector<Quad> localQuads(quads);
	vector<Quad> orderedQuads;

	// Third, take the topmost quad. Find everything in its row.
	// order from left to right. Number them, remove them
	int quadNumber = 1;
	int iteration = 0;
	while (!localQuads.empty())
	{
		if (quadNumber > 200)
		{
			break;
		}

		// Get the top quad, remove it
		Quad topQuad;
		topQuad.centre = Point(100000,100000); // obviously not the top Quad
		int topIndex = 0;
		for (int i = 0; i < localQuads.size(); ++i)
		{
			Quad& q = localQuads[i];
			if (q.centre.y < topQuad.centre.y)
			{
				topQuad = q;
				topIndex = i;
			}
		}
		vector<Quad> thisRow;
		thisRow.push_back(topQuad);
		localQuads.erase(localQuads.begin() + topIndex);

		// Get margin of error
		int margin = L2norm(topQuad.centre - topQuad.points[0]);

		// Find all quads in this row and remove them
		while (true)
		{
			// Find first available quad in this row
			bool found = false;
			int i = 0;
			for (i = 0; i < localQuads.size(); ++i)
			{
				Quad& q = localQuads[i];
				if (abs(q.centre.y - topQuad.centre.y) < margin / 2 && q.centre != topQuad.centre)
				{
					thisRow.push_back(q);
					found = true;
					break;
				}
			}

			// We found all the quads in this row that we could
			if (!found)
			{
				break;
			}

			// Remove the quad
			localQuads.erase(localQuads.begin() + i);

			if (localQuads.empty())
			{
				break;
			}
		}

		// Order quads
		sort(thisRow.begin(), thisRow.end(), OrderTwoQuadsByAscendingCentreX);

		// Number quads
		for (unsigned int n = 0; n < thisRow.size(); ++n)
		{
			thisRow[n].number = quadNumber;
			quadNumber++;
		}

		// What if a row is too small? Gotta account
		switch (iteration)
		{
		case 0: 
			quadNumber = quadNumber < 6 ? 6 : quadNumber;
			break;
		case 1:
			quadNumber = quadNumber < 10 ? 10 : quadNumber;
			break;
		case 2:
			quadNumber = quadNumber < 15 ? 15 : quadNumber;
			break;
		case 3:
			quadNumber = quadNumber < 19 ? 19 : quadNumber;
			break;
		case 4:
			quadNumber = quadNumber < 24 ? 24 : quadNumber;
			break;
		case 5:
			quadNumber = quadNumber < 29 ? 29 : quadNumber;
			break;
		default:
			break;
		}

		// Add to ordered list
		for (Quad& q : thisRow)
		{
			orderedQuads.push_back(q);
		}

		iteration++;
	}

	// Repeat
	quads.clear();
	for (Quad& q : orderedQuads)
	{
		quads.push_back(q);
	}
}

/*
	Get the intrinsic and extrinsic parameters from the homography

	// Decompose into K matrix and extrinsics
	// upper triangular numpty
	// perform LDLT decomposition
	// normalise first L, that's K
	// This is because K is already upper triangular

	// DLT is the homography
	// Make sure r1 and r2 are orthogonal
*/
bool ComputeIntrinsicsAndExtrinsicFromHomography(const Matrix3f& H, Matrix3f& K, Matrix3f& T)
{
	LDLT<Matrix3f> ldlt(3);

	ldlt.compute(H);

	auto L = ldlt.matrixL();
	K = L;
	auto DLT = K.inverse() * H;

	// Now to get the homography
	// So we have a three-by-three for rotation, except one of the rotation vectors
	// is irrelevant and I've forgotten why, and therefor the last bit is the translation

	// Set T = DL^T
    // Check that the first two columns are orthogonal
	T = DLT;
	Vector3f r0 = T.col(0);
	Vector3f r1 = T.col(1);

	// r0 dot r1 should be 0
	if (r0.dot(r1) != 0)
	{
		cout << "Rotation vectors are not orthogonal!" << endl;
		return false;
	}

	return true;
}