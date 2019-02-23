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
//#define DEBUG_CORNERS

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
	// downsample for this
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
		// use downsampled version

#ifdef DEBUG
		namedWindow("erode", WINDOW_NORMAL);
		imshow("erode", erode);
		if (debug) waitKey(0);
#endif

		// Somehow go high res here?

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
				q.size = 0;
				for (int idx = 0; idx < 4; ++idx)
				{
					q.size += DistBetweenPoints(q.centre, q.points[idx])/4;
				}

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
							//DrawQuad(orig, q2);
							//DrawQuad(newErode, q1);
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
					//if (debug) DrawQuad(newErode, q1);
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

	auto temp5 = checkerboard.clone();
	for (Quad q : quads)
	{
		rectangle(temp5, q.points[0], q.centre, (128, 128, 128), CV_FILLED);
		rectangle(temp5, q.points[1], q.centre, (128, 128, 128), CV_FILLED);
		rectangle(temp5, q.points[2], q.centre, (128, 128, 128), CV_FILLED);
		rectangle(temp5, q.points[3], q.centre, (128, 128, 128), CV_FILLED);
	}


	// Debug display
	imshow("all quads", temp5);
	waitKey(0);
	

	// Link corners
	// For each pair of quads, find any corners they share
	// Note these links in an array, where the index of the quad's own corner
	// holds a pair of the ID of the other quad, plus the corner index it links to
	for (int i = 0; i < quads.size(); ++i)
	{
		Quad& q1 = quads[i];

		if (q1.numLinkedCorners == 4)
		{
			// Can't have nay more corners
			continue;
		}

		// Find all quads within a certain radius
		vector<pair<int, Quad>> closestQuads;
		const float diag1 = GetLongestDiagonal(q1);
		for (int j = i + 1; j < quads.size(); ++j)
		{
			Quad& q2 = quads[j];

			if (q2.numLinkedCorners == 4)
			{
				continue;
			}

			// Sanity check - if their centres are further away than twice the longest diagonal of the first quad, 
			// ignore this quad
			if (DistBetweenPoints(q1.centre, q2.centre) < 2 * diag1)
			{
				closestQuads.push_back(pair<int, Quad>(j, q2));
			}	
		}

		// For each corner of q1, find the closest point amongst the closest quads
		for (int c = 0; c < 4; ++c)
		{
			Point corner = q1.points[c];
			float minDistToPoint = 2*diag1; // upper bound

			int closestQuadIndex = 0;
			int closestPointIndex = 0;
			for (int k = 0; k < closestQuads.size(); ++k)
			{
				Quad q2 = closestQuads[k].second;
				for (int c2 = 0; c2 < 4; ++c2)
				{
					float d = DistBetweenPoints(q2.points[c2], corner);
					if (d < minDistToPoint)
					{
						minDistToPoint = d;
						closestPointIndex = c2;
						closestQuadIndex = closestQuads[k].first;
					}
				}
			}

			Quad& q2 = quads[closestQuadIndex];
			Point corner2 = q2.points[closestPointIndex];

			if (debug)
			{
				auto temp = checkerboard.clone();
				rectangle(temp, q1.points[0], q1.centre, (128, 128, 128), CV_FILLED);
				rectangle(temp, q1.points[1], q1.centre, (128, 128, 128), CV_FILLED);
				rectangle(temp, q1.points[2], q1.centre, (128, 128, 128), CV_FILLED);
				rectangle(temp, q1.points[3], q1.centre, (128, 128, 128), CV_FILLED);
				rectangle(temp, q2.points[0], q2.centre, (128, 128, 128), CV_FILLED);
				rectangle(temp, q2.points[1], q2.centre, (128, 128, 128), CV_FILLED);
				rectangle(temp, q2.points[2], q2.centre, (128, 128, 128), CV_FILLED);
				rectangle(temp, q2.points[3], q2.centre, (128, 128, 128), CV_FILLED);

				circle(temp, q1.centre, 2*diag1, (128, 128, 128), 2);

				// Debug display
				imshow("The two quads under consideration", temp);
				waitKey(0);
			}

			// Do these corners lie within a rectangle defined by the centres?
			/*
			|------c2
			|	    |
			|       |
			c1------|
			*/
			if (!DoesPointLieWithinQuadOfTwoCentres(corner, q1, q2))
			{
				// This corner failed to match. Move on
				continue;
			}
			if (!DoesPointLieWithinQuadOfTwoCentres(corner2, q1, q2))
			{
				// This corner failed to match. Move on
				continue;
			}

			// Have these points already matched before?
			if (q2.associatedCorners[closestPointIndex].first != -1)
			{
				if (q2.associatedCorners[closestPointIndex].first != q1.id)
				{
					// Two quads have matched to the same corner.
					// Assume the first is right
					continue;
				}
			}
			if (q1.associatedCorners[c].first != -1)
			{
				if (q1.associatedCorners[c].first != q2.id)
				{
					// Two quads have matched to the same corner.
					// Assume the first is right
					continue;
				}
			}

			if (DistBetweenPoints(corner, corner2) > 0.7*diag1)
			{
				continue;
			}

			// Things worked! Mark this
			Point cornerFinal((corner.x + corner2.x) / 2, (corner.y + corner2.y) / 2);
			q1.points[c] = cornerFinal;
			q2.points[closestPointIndex] = cornerFinal;
			q1.associatedCorners[c] = pair<int, int>(q2.id, closestPointIndex);
			q2.associatedCorners[closestPointIndex] = pair<int, int>(q1.id, c);
			q1.numLinkedCorners++;
			q2.numLinkedCorners++;

			if (debug)
			{
				auto temp = checkerboard.clone();
				rectangle(temp, q1.points[0], q1.centre, (128, 128, 128), 1);
				rectangle(temp, q1.points[1], q1.centre, (128, 128, 128), 1);
				rectangle(temp, q1.points[2], q1.centre, (128, 128, 128), 1);
				rectangle(temp, q1.points[3], q1.centre, (128, 128, 128), 1);
				rectangle(temp, q2.points[0], q2.centre, (128, 128, 128), 1);
				rectangle(temp, q2.points[1], q2.centre, (128, 128, 128), 1);
				rectangle(temp, q2.points[2], q2.centre, (128, 128, 128), 1);
				rectangle(temp, q2.points[3], q2.centre, (128, 128, 128), 1);

				rectangle(temp, q1.centre, q2.centre, (128, 128, 128), 1);

				// Debug display
				imshow("It worked", temp);
				waitKey(0);
			}
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
// just do reprojection error from the corner centres
float GetReprojectionError(const Quad gtCorners[], const Quad corners[], int indices[], const Matrix3f& H)
{

	// Find the closest quad in gt set and get error
	float e = 0;
	for (int i = 0; i < 4; ++i)
	{
		Quad q1 = gtCorners[i];
		Quad q2 = corners[indices[i]];

		Vector3f x(q2.centre.x, q2.centre.y, 1);
		Vector3f Hx = H * x;
		Hx / Hx(2);
		auto newQ2centre = Point(Hx(0), Hx(1));

		e += L2norm(q1.centre - newQ2centre);
	}

	return e;
}
// Actual Function
bool GetHomographyAndMatchQuads(Matrix3f& H, const Mat& img, const cv::Mat& checkerboard, vector<Quad>& gtQuads, vector<Quad>& quads)
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
	Quad gtCorners[4] = {topleft, topright, bottomleft, bottomright};

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
	imshow("corners in captured", temp2);
	waitKey(0);

	Mat temp3 = checkerboard.clone();
	rectangle(temp3, topleft.points[0], topleft.centre, (128, 128, 128), 1);
	rectangle(temp3, topleft.points[1], topleft.centre, (128, 128, 128), 1);
	rectangle(temp3, topleft.points[2], topleft.centre, (128, 128, 128), 1);
	rectangle(temp3, topleft.points[3], topleft.centre, (128, 128, 128), 1);
	rectangle(temp3, topright.points[0], topright.centre, (128, 128, 128), 1);
	rectangle(temp3, topright.points[1], topright.centre, (128, 128, 128), 1);
	rectangle(temp3, topright.points[2], topright.centre, (128, 128, 128), 1);
	rectangle(temp3, topright.points[3], topright.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomleft.points[0], bottomleft.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomleft.points[1], bottomleft.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomleft.points[2], bottomleft.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomleft.points[3], bottomleft.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomright.points[0], bottomright.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomright.points[1], bottomright.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomright.points[2], bottomright.centre, (128, 128, 128), 1);
	rectangle(temp3, bottomright.points[3], bottomright.centre, (128, 128, 128), 1);
	// Debug display
	imshow("corners in gt", temp3);
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
			Mat temp3 = checkerboard.clone();
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



				circle(temp3, centre, 20, (128, 128, 128), -1);
			} 
			// Debug display
			imshow("permutation", temp3);
			waitKey(0);


			// Transform all captured quads with H

			float e = GetReprojectionError(gtCorners, corners, indices, h);
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

		corners[perm[0]].number = 1;
		corners[perm[1]].number = 5;
		corners[perm[2]].number = 28;
		corners[perm[3]].number = 32;
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

	Mat temp6 = checkerboard.clone();
	for (Quad q : quads)
	{

		Vector3f x(q.centre.x, q.centre.y, 1);
		Vector3f Hx = H * x;
		Hx / Hx(2);
		auto centre = Point(Hx(0), Hx(1));



		circle(temp6, centre, 20, (128, 128, 128), -1);
	}
	// Debug display
	imshow("Final homography", temp6);
	waitKey(0);

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

	// Can we coerce these into being orthogonal?

	// r0 dot r1 should be 0
	if (r0.dot(r1) != 0)
	{
		cout << "Rotation vectors are not orthogonal!" << endl; // not yet
		//return false;
	}

	return true;
}