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
				q.number = 0;
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
float GetReprojectionError(const Quad gtCorners[], const vector<Quad> corners, int indices[], const Matrix3f& H)
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
		auto newQ2centre = Point2f(Hx(0), Hx(1));

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

	// TODO: make these indices

	Quad topleft = gtQuads[0];
	Quad topright = gtQuads[0];
	Quad bottomleft = gtQuads[0];
	Quad bottomright = gtQuads[0];
	for (Quad& q : gtQuads)
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
	vector<Quad> corners;
	for (const Quad& q : quads)
	{
		if (q.numLinkedCorners == 1)
		{
			corners.push_back(q);
			if (corners.size() == 4) break;
		}
	}

	Mat temp2 = img.clone();
	for (int i = 0; i < corners.size(); ++i)
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

	if (corners.size() == 4)
	{
		// normalise all the points
		topleft.centre.x /= (float)checkerboard.cols;
		topleft.centre.y /= (float)checkerboard.rows;
		topright.centre.x /= (float)checkerboard.cols;
		topright.centre.y /= (float)checkerboard.rows;
		bottomright.centre.x /= (float)checkerboard.cols;
		bottomright.centre.y /= (float)checkerboard.rows;
		bottomleft.centre.x /= (float)checkerboard.cols;
		bottomleft.centre.y /= (float)checkerboard.rows;

		// TODO: accept floats in points

		// We have all four corners.
		// Order them in increasing angle relative to horizontal from centre
		Point centre(img.cols/2, img.rows/2);
		corners[0].angleToCentre = atan2(centre.y - corners[0].centre.y, corners[0].centre.x - centre.x) * 180 / PI;
		corners[1].angleToCentre = atan2(centre.y - corners[1].centre.y, corners[1].centre.x - centre.x) * 180 / PI;
		corners[2].angleToCentre = atan2(centre.y - corners[2].centre.y, corners[2].centre.x - centre.x) * 180 / PI;
		corners[3].angleToCentre = atan2(centre.y - corners[3].centre.y, corners[3].centre.x - centre.x) * 180 / PI;
		sort(corners.begin(), corners.end(), CompareQuadByAngleToCentre);

		// This gives us only four possibilities, of which only two should work
		corners[0].centre.x = (float)corners[0].centre.x/(float)img.cols;
		corners[0].centre.y /= (float)img.rows;
		corners[1].centre.x /= (float)img.cols;
		corners[1].centre.y /= (float)img.rows;
		corners[2].centre.x /= (float)img.cols;
		corners[2].centre.y /= (float)img.rows;
		corners[3].centre.x /= (float)img.cols;
		corners[3].centre.y /= (float)img.rows;

		// Test all 24 possibilities for minimal reprojection error
		

		// Iterate over all permutations
		float minError = 100000000;
		int perm[] = { 0,1,2,3 };
		Matrix3f homography;
		for (int i = 0; i < 4; ++i)
		{

		//}
		//do {
			// the gt corners 0,1,2,3 are associated with indices[0],[1],[2],[3] of captured corners
			vector<pair<Point2f, Point2f>> matches;
			matches.push_back(pair<Point2f, Point2f>(corners[i].centre, topleft.centre));
			matches.push_back(pair<Point2f, Point2f>(corners[(i+1)%4].centre, topright.centre));
			matches.push_back(pair<Point2f, Point2f>(corners[(i+2)%4].centre, bottomleft.centre));
			matches.push_back(pair<Point2f, Point2f>(corners[(i+3)%4].centre, bottomright.centre));

			cout << "Matches this round are: " << endl;
			for (auto m : matches)
			{
				cout << "\t" << m.first << " <-> " << m.second << endl;
				
			}

			Matrix3f h;
			if (!GetHomographyFromMatches(matches, h))
			{
				// This permutation wasn't good enough
				continue;
			}

			// DEBUG
			// Draw the homographied quads
			Mat temp3 = checkerboard.clone(); // checkerboard
			for (Quad q : quads) // quads
			{
				Vector3f x(q.centre.x/(float)img.cols, q.centre.y / (float)img.rows, 1);
				Vector3f Hx = h * x;
				Hx /= Hx(2);
				auto centre = Point2f(Hx(0), Hx(1));

				centre.x *= (float)temp3.cols;
				centre.y *= (float)temp3.rows;

				circle(temp3, Point((int)centre.x, (int)centre.y), 20, (128, 128, 128), CV_FILLED);
			} 
			// Debug display
			imshow("homography", temp3);
			waitKey(0);
			


			// NOW FIX THIS
			// Update reprojection error

			// Transform all captured quads with H
			int indices[] = { i,(i + 1) % 4,(i + 2) % 4,(i + 3) % 4 };
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

		}// while (std::next_permutation(indices, indices + 4));

		if (minError == 100000000)
		{
			// Nothing worked. Just bail
			return false;
		}

		cout << "The minimum error was " << minError << endl;

		H = homography;

		for (Quad& q : quads)
		{
			if (q.id == corners[perm[0]].id)
			{
				q.number = 1;
			}
			if (q.id == corners[perm[1]].id)
			{
				q.number = 5;
			}
			if (q.id == corners[perm[2]].id)
			{
				q.number = 28;
			}
			if (q.id == corners[perm[3]].id)
			{
				q.number = 32;
			}
		}
		//corners[perm[0]].number = 1;
		//corners[perm[1]].number = 5;
		//corners[perm[2]].number = 28;
		//corners[perm[3]].number = 32;
	}
	else if (corners.size() == 3 || corners.size() == 2)
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
		// NEED TO NORMALISE
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
	int currentQuadIndex = 1;

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

	// Do each row separately, hardcoded
	// Horrible, but hey it works. 

	// Find quads #1 and #5:
	int indexQ1 = -1;
	int indexQ5 = -1;
	for (int i = 0; i < quads.size(); ++i)
	{
		if (quads[i].number == 1)
		{
			indexQ1 = i;
		}
		if (quads[i].number == 5)
		{
			indexQ5 = i;
		}
	}
	if (indexQ1 == -1 || indexQ5 == -1)
	{
		// we failed? Return. SHould have an error code. At least false?
		return;
	}
	
	Quad& q1 = quads[indexQ1];
	Quad& q5 = quads[indexQ5];
	// Draw a line between the two
	LineSegment l;
	l.p1 = q1.centre;
	l.p2 = q5.centre;

	// Find the three other quads whose centres lie within half a diagonal's length of the line
	float bound = GetLongestDiagonal(q1)/2;
	vector<Quad> quadsInRow; // TODO
	for (Quad& q : quads)
	{
		float d = abs(PointDistToLineSigned(q.centre, q1.centre, q5.centre));
		if (d < bound)
		{
			quadsInRow.push_back(q);
		}
	}

	// Order all these quads by x coord
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i].number = currentQuadIndex;
		currentQuadIndex++;
	}
	q5.number = 5;
	currentQuadIndex = 6;

	// Number the two attached to q1 and q5
	int indexQ6 = 0;
	int indexQ9 = 0;
	for (int i = 0; i < 4; ++i)
	{
		if (q1.associatedCorners[i].first != -1)
		{
			quads[q1.associatedCorners[i].second].number = 6;
			indexQ6 = i;
		}
		if (q5.associatedCorners[i].first != -1)
		{
			quads[q5.associatedCorners[i].second].number = 9;
			indexQ9 = i;
		}
	}

	// Next row
	Quad& q6 = quads[q1.associatedCorners[indexQ6].second];
	Quad& q9 = quads[q5.associatedCorners[indexQ9].second];
	// Draw a line between the two
	l.p1 = q6.centre;
	l.p2 = q9.centre;

	bound = GetLongestDiagonal(q6) / 2;
	quadsInRow.clear();
	for (Quad& q : quads)
	{
		float d = abs(PointDistToLineSigned(q.centre, q6.centre, q9.centre));
		if (d < bound)
		{
			quadsInRow.push_back(q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i].number = currentQuadIndex;
		currentQuadIndex++;
	}
	q9.number = 9;
	currentQuadIndex = 10;

	// Number the two attached to q1 and q5
	int indexQ10 = -1;
	int indexQ14 = -1;
	for (int i = 0; i < 4; ++i)
	{
		if (q6.associatedCorners[i].first != -1)
		{
			if (quads[q6.associatedCorners[i].second].number == 0)
			{
				if (quads[q6.associatedCorners[i].second].centre.y > q6.centre.y && quads[q6.associatedCorners[i].second].centre.x < q6.centre.x)
				{
					quads[q6.associatedCorners[i].second].number = 10;
					indexQ10 = i;
				}
			}
		}
		if (q9.associatedCorners[i].first != -1)
		{
			if (quads[q9.associatedCorners[i].second].number == 0)
			{
				if (quads[q9.associatedCorners[i].second].centre.y > q9.centre.y)
				{
					quads[q9.associatedCorners[i].second].number = 14;
					indexQ14 = i;
				}
			}
		}
	}
	if (indexQ10 == -1 || indexQ14 == -1)
	{
		// we failed? Return. SHould have an error code. At least false?
 		return;
	}

	// Next row
	Quad& q10 = quads[q1.associatedCorners[indexQ10].second];
	Quad& q14 = quads[q5.associatedCorners[indexQ14].second];
	// Draw a line between the two
	l.p1 = q10.centre;
	l.p2 = q14.centre;

	bound = GetLongestDiagonal(q10) / 2;
	quadsInRow.clear();
	for (Quad& q : quads)
	{
		float d = abs(PointDistToLineSigned(q.centre, q10.centre, q14.centre));
		if (d < bound)
		{
			quadsInRow.push_back(q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i].number = currentQuadIndex;
		currentQuadIndex++;
	}
	q14.number = 14;
	currentQuadIndex = 15;

	// Number the two attached to q1 and q5
	int indexQ15 = -1;
	int indexQ18 = -1;
	for (int i = 0; i < 4; ++i)
	{
		if (q10.associatedCorners[i].first != -1)
		{
			if (quads[q10.associatedCorners[i].second].number == 0)
			{
				if (quads[q10.associatedCorners[i].second].centre.y > q10.centre.y)
				{
					quads[q10.associatedCorners[i].second].number = 15;
					indexQ15 = i;
				}
			}
		}
		if (q14.associatedCorners[i].first != -1)
		{
			if (quads[q14.associatedCorners[i].second].number == 0)
			{
				if (quads[q14.associatedCorners[i].second].centre.y > q14.centre.y)
				{
					quads[q14.associatedCorners[i].second].number = 18;
					indexQ18 = i;
				}
			}
		}
	}
	if (indexQ15 == -1 || indexQ18 == -1)
	{
		// we failed? Return. SHould have an error code. At least false?
		return;
	}

	// Next row
	Quad& q15 = quads[q1.associatedCorners[indexQ15].second];
	Quad& q18 = quads[q5.associatedCorners[indexQ18].second];
	// Draw a line between the two
	l.p1 = q15.centre;
	l.p2 = q18.centre;

	bound = GetLongestDiagonal(q15) / 2;
	quadsInRow.clear();
	for (Quad& q : quads)
	{
		float d = abs(PointDistToLineSigned(q.centre, q15.centre, q18.centre));
		if (d < bound)
		{
			quadsInRow.push_back(q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i].number = currentQuadIndex;
		currentQuadIndex++;
	}
	q18.number = 18;
	currentQuadIndex = 19;

	// Number the two attached to q1 and q5
	int indexQ19 = -1;
	int indexQ23 = -1;
	for (int i = 0; i < 4; ++i)
	{
		if (q15.associatedCorners[i].first != -1)
		{
			if (quads[q15.associatedCorners[i].second].number == 0)
			{
				if (quads[q15.associatedCorners[i].second].centre.y > q15.centre.y)
				{
					quads[q15.associatedCorners[i].second].number = 19;
					indexQ19 = i;
				}
			}
		}
		if (q18.associatedCorners[i].first != -1)
		{
			if (quads[q18.associatedCorners[i].second].number == 0)
			{
				if (quads[q18.associatedCorners[i].second].centre.y > q18.centre.y)
				{
					quads[q18.associatedCorners[i].second].number = 23;
					indexQ23 = i;
				}
			}
		}
	}
	if (indexQ19 == -1 || indexQ23 == -1)
	{
		// we failed? Return. SHould have an error code. At least false?
		return;
	}

	// Next row
	Quad& q19 = quads[q1.associatedCorners[indexQ19].second];
	Quad& q23 = quads[q5.associatedCorners[indexQ23].second];
	// Draw a line between the two
	l.p1 = q19.centre;
	l.p2 = q23.centre;

	bound = GetLongestDiagonal(q19) / 2;
	quadsInRow.clear();
	for (Quad& q : quads)
	{
		float d = abs(PointDistToLineSigned(q.centre, q19.centre, q23.centre));
		if (d < bound)
		{
			quadsInRow.push_back(q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i].number = currentQuadIndex;
		currentQuadIndex++;
	}
	q23.number = 23;
	currentQuadIndex = 24;

	// Number the two attached to q1 and q5
	int indexQ24 = -1;
	int indexQ27 = -1;
	for (int i = 0; i < 4; ++i)
	{
		if (q19.associatedCorners[i].first != -1)
		{
			if (quads[q19.associatedCorners[i].second].number == 0)
			{
				if (quads[q19.associatedCorners[i].second].centre.y > q19.centre.y)
				{
					quads[q19.associatedCorners[i].second].number = 24;
					indexQ24 = i;
				}
			}
		}
		if (q23.associatedCorners[i].first != -1)
		{
			if (quads[q23.associatedCorners[i].second].number == 0)
			{
				if (quads[q23.associatedCorners[i].second].centre.y > q23.centre.y)
				{
					quads[q23.associatedCorners[i].second].number = 27;
					indexQ27 = i;
				}
			}
		}
	}
	if (indexQ24 == -1 || indexQ27 == -1)
	{
		// we failed? Return. SHould have an error code. At least false?
		return;
	}

	// Next row
	Quad& q24 = quads[q1.associatedCorners[indexQ24].second];
	Quad& q27 = quads[q5.associatedCorners[indexQ27].second];
	// Draw a line between the two
	l.p1 = q24.centre;
	l.p2 = q27.centre;

	bound = GetLongestDiagonal(q24) / 2;
	quadsInRow.clear();
	for (Quad& q : quads)
	{
		float d = abs(PointDistToLineSigned(q.centre, q24.centre, q27.centre));
		if (d < bound)
		{
			quadsInRow.push_back(q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i].number = currentQuadIndex;
		currentQuadIndex++;
	}
	q27.number = 27;
	currentQuadIndex = 28;

	// Number the two attached to q1 and q5
	int indexQ28 = -1;
	int indexQ32 = -1;
	for (int i = 0; i < 4; ++i)
	{
		if (q24.associatedCorners[i].first != -1)
		{
			if (quads[q24.associatedCorners[i].second].number == 28)
			{
				indexQ28 = i;
			}
		}
		if (q18.associatedCorners[i].first != -1)
		{
			if (quads[q18.associatedCorners[i].second].number == 32)
			{
				indexQ32 = i;
			}
		}
	}
	if (indexQ28 == -1 || indexQ32 == -1)
	{
		// we failed? Return. SHould have an error code. At least false?
		return;
	}

	// Final row
	Quad& q28 = quads[q1.associatedCorners[indexQ28].second];
	Quad& q32 = quads[q5.associatedCorners[indexQ32].second];
	// Draw a line between the two
	l.p1 = q28.centre;
	l.p2 = q32.centre;

	bound = GetLongestDiagonal(q28) / 2;
	quadsInRow.clear();
	for (Quad& q : quads)
	{
		float d = abs(PointDistToLineSigned(q.centre, q28.centre, q32.centre));
		if (d < bound)
		{
			quadsInRow.push_back(q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i].number = currentQuadIndex;
		currentQuadIndex++;
	}
	q32.number = 32;
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