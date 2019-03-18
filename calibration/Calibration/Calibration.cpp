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
float L2norm(Point2f a)
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
#ifdef DEBUG
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
#endif
	

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
float GetReprojectionError(const vector<Quad>& gtQuads, const vector<Quad>& quads,
	                       const Quad gtCorners[], const Point2f gtSize,
	                       const Point2f size, const vector<Quad> corners, 
	                       vector<int> indices, const Matrix3f& H)
{

	// Find the closest quad in gt set and get error
	float e = 0;
	for (int i = 0; i < 4; ++i)
	{
		Quad q1 = gtCorners[i];
		Quad q2 = corners[indices[i]];

		Vector3f x(q2.centre.x, q2.centre.y, 1);
		Vector3f Hx = H * x;
		Hx /= Hx(2);
		// Put everything into the same coordinates
		auto newQ2centre = Point2f(Hx(0)*gtSize.x, Hx(1)*gtSize.y);
		auto newQ1centre = Point2f(q1.centre.x*gtSize.x, q1.centre.y*gtSize.y);

		// This is in normalise
		e += L2norm(newQ1centre - newQ2centre);

		// INclude in the reprojection error the difference between the quads
		// that each of these connect to
		Quad q1_1, q2_1;
		for (int j = 0; j < 4; ++j)
		{
			if (q1.associatedCorners[j].second != -1)
			{
				q1_1 = gtQuads[q1.associatedCorners[j].second];
			}
			if (q2.associatedCorners[j].second != -1)
			{
				q2_1 = quads[q2.associatedCorners[j].second];
			}
		}

		Vector3f x2(q2_1.centre.x/size.x, q2_1.centre.y/size.y, 1);
		Vector3f Hx2 = H * x2;
		Hx2 /= Hx2(2);
		auto newQ1_1centre = Point2f(Hx2(0)*gtSize.x, Hx2(1)*gtSize.y);

		e += L2norm(q1_1.centre - newQ1_1centre);
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
	for (Quad q : gtQuads)
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
	Quad gtCorners[4] = {topleft, topright, bottomright, bottomleft};

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

	if (corners.size() == 4)
	{
		// We have all four corners.
		// Order them in increasing angle relative to horizontal from centre
		Point centre(img.cols/2, img.rows/2);
		corners[0].angleToCentre = atan2(centre.y - corners[0].centre.y, corners[0].centre.x - centre.x) * 180 / PI;
		corners[1].angleToCentre = atan2(centre.y - corners[1].centre.y, corners[1].centre.x - centre.x) * 180 / PI;
		corners[2].angleToCentre = atan2(centre.y - corners[2].centre.y, corners[2].centre.x - centre.x) * 180 / PI;
		corners[3].angleToCentre = atan2(centre.y - corners[3].centre.y, corners[3].centre.x - centre.x) * 180 / PI;

		// get them clockwise, not anticlockwise
		sort(corners.begin(), corners.end(), CompareQuadByAngleToCentre);

		// This gives us only four possibilities, of which only two should work
		// Normalise the points for the homography matching
		for (int j = 0; j < 4; ++j)
		{
			corners[j].centre.x /= (float)img.cols;
			corners[j].centre.y /= (float)img.rows;

			gtCorners[j].centre.x /= (float)checkerboard.cols;
			gtCorners[j].centre.y /= (float)checkerboard.rows;
		}

		// Iterate over all permutations
		float minError = 100000000;
		int perm[] = { 0,1,2,3 };
		Matrix3f homography;
		for (int i = 0; i < 4; ++i)
		{
			vector<pair<Point2f, Point2f>> matches;
			matches.push_back(pair<Point2f, Point2f>(corners[i].centre, gtCorners[0].centre));
			matches.push_back(pair<Point2f, Point2f>(corners[(i+1)%4].centre, gtCorners[1].centre));
			matches.push_back(pair<Point2f, Point2f>(corners[(i+2)%4].centre, gtCorners[2].centre));
			matches.push_back(pair<Point2f, Point2f>(corners[(i+3)%4].centre, gtCorners[3].centre));

			Matrix3f h;
			if (!GetHomographyFromMatches(matches, h))
			{
				// This permutation wasn't good enough
				continue;
			}

			// How mcuh error did this iteration get?
			vector<int> indices = { i,(i + 1) % 4,(i + 2) % 4,(i + 3) % 4 };
			float e = GetReprojectionError(gtQuads, quads, gtCorners, Point2f(checkerboard.cols, checkerboard.rows), Point2f(img.cols, img.rows), corners, indices, h);
  			if (e < minError)
			{
				minError = e;
				perm[0] = indices[0];
				perm[1] = indices[1];
				perm[2] = indices[2];
				perm[3] = indices[3];
				homography = h;
			}
		}

		if (minError == 100000000)
		{
			// Nothing worked. Just bail
			return false;
		}

#ifdef DEBUG
		cout << "The minimum error was " << minError << endl;
#endif
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

#ifdef DEBUG
	Mat temp6 = checkerboard.clone();
	for (Quad q : quads)
	{
		Vector3f x(q.centre.x / (float)img.cols, q.centre.y / (float)img.rows, 1);
		Vector3f Hx = H * x;
		Hx /= Hx(2);
		auto centre = Point2f(Hx(0), Hx(1));

		centre.x *= (float)temp6.cols;
		centre.y *= (float)temp6.rows;

		if (!IsInBounds(temp6.rows, temp6.cols, centre))
		{
			continue;
		}

		circle(temp6, centre, 20, (128, 128, 128), -1);
	}
	// Debug display
	imshow("Final homography", temp6);
	waitKey(0);
#endif

	// We just need the smallest projection error, really

	// Now number quads
	//TransformAndNumberQuads(H, Point2f(checkerboard.cols, checkerboard.rows), Point2f(img.cols, img.rows), quads);

	return true;
}

/*
	Transform and number quads

	H is expected to transform the plane the quads are in to the ground truth plane
	where is it a trivial matter to scan through and assign the correct number to each quad

	We use a bound of half the transformed top left quad's diagonal as the measure for whether
	or not another quad is in the same column or row

*/
void TransformAndNumberQuads(const Eigen::Matrix3f& H, const Point2f gtSize, const Point2f size, std::vector<Quad>& quads)
{
	int currentQuadIndex = 1;

	// First, transform all quads
	for (Quad& q : quads)
	{
		Vector3f x(q.centre.x / size.x, q.centre.y / size.y, 1);
		Vector3f Hx = H * x;
		Hx /= Hx(2);
		q.centre = Point2f(Hx(0)*gtSize.x, Hx(1)*gtSize.y);
	}

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
	vector<Quad*> quadsInRow;
	// make this a list of indices
	for (int i = 0; i < quads.size(); ++i)
	{
		float d = abs(PointDistToLineSigned(quads[i].centre, q1.centre, q5.centre));
		if (d < bound)
		{
			quadsInRow.push_back(&quads[i]);
		}
	}

	// Order all these quads by x coord
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i]->number = currentQuadIndex;
		currentQuadIndex++;
	}
	q5.number = 5;
	currentQuadIndex = 6;

	// Number the two attached to q1 and q5
	int indexQ6 = -1;
	int indexQ9 = -1;
	for (int n = 0; n < quads.size(); ++n)
	{
		for (int i = 0; i < 4; ++i)
		{
			if (q1.associatedCorners[i].first != -1)
			{
				if (q1.associatedCorners[i].first == quads[n].id)
					indexQ6 = n;
			}
			if (q5.associatedCorners[i].first != -1)
			{
				if (q5.associatedCorners[i].first == quads[n].id)
					indexQ9 = n;
			}
		}
	}

	// Precautionary
	if (indexQ6 == -1 || indexQ9 == -1)
	{
		return;
	}

	// Next row
	Quad& q6 = quads[indexQ6];
	Quad& q9 = quads[indexQ9];
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
			quadsInRow.push_back(&q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i]->number = currentQuadIndex;
		currentQuadIndex++;
	}
	q9.number = 9;
	currentQuadIndex = 10;

	// Number the two attached to q1 and q5
	int indexQ10 = -1;
	int indexQ14 = -1;
	for (int n = 0; n < quads.size(); ++n)
	{
		for (int i = 0; i < 4; ++i)
		{
			if (q6.associatedCorners[i].first != -1)
			{
				if (q6.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q6.centre.y && quads[n].centre.x < q6.centre.x)
					{
						quads[n].number = 10;
						indexQ10 = n;
					}
				}
			}
			if (q9.associatedCorners[i].first != -1)
			{
				if (q9.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q9.centre.y  && quads[n].centre.x > q6.centre.x)
					{
						quads[n].number = 14;
						indexQ14 = n;
					}
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
	Quad& q10 = quads[indexQ10];
	Quad& q14 = quads[indexQ14];
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
			quadsInRow.push_back(&q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i]->number = currentQuadIndex;
		currentQuadIndex++;
	}
	q14.number = 14;
	currentQuadIndex = 15;

	// Number the two attached to q1 and q5
	int indexQ15 = -1;
	int indexQ18 = -1;
	for (int n = 0; n < quads.size(); ++n)
	{
		for (int i = 0; i < 4; ++i)
		{
			if (q10.associatedCorners[i].first != -1)
			{
				if (q10.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q10.centre.y)
					{
						quads[n].number = 15;
						indexQ15 = n;
					}
				}
			}
			if (q14.associatedCorners[i].first != -1)
			{
				if (q14.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q14.centre.y)
					{
						quads[n].number = 18;
						indexQ18 = n;
					}
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
	Quad& q15 = quads[indexQ15];
	Quad& q18 = quads[indexQ18];
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
			quadsInRow.push_back(&q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i]->number = currentQuadIndex;
		currentQuadIndex++;
	}
	q18.number = 18;
	currentQuadIndex = 19;

	// Number the two attached to q1 and q5
	int indexQ19 = -1;
	int indexQ23 = -1;
	for (int n = 0; n < quads.size(); ++n)
	{
		for (int i = 0; i < 4; ++i)
		{
			if (q15.associatedCorners[i].first != -1)
			{
				if (q15.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q15.centre.y && quads[n].centre.x < q15.centre.x)
					{
						quads[n].number = 19;
						indexQ19 = n;
					}
				}
			}
			if (q18.associatedCorners[i].first != -1)
			{
				if (q18.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q18.centre.y && quads[n].centre.x > q18.centre.x)
					{
						quads[n].number = 23;
						indexQ23 = n;
					}
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
	Quad& q19 = quads[indexQ19];
	Quad& q23 = quads[indexQ23];
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
			quadsInRow.push_back(&q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i]->number = currentQuadIndex;
		currentQuadIndex++;
	}
	q23.number = 23;
	currentQuadIndex = 24;

	// Number the two attached to q1 and q5
	int indexQ24 = -1;
	int indexQ27 = -1;
	for (int n = 0; n < quads.size(); ++n)
	{
		for (int i = 0; i < 4; ++i)
		{
			if (q19.associatedCorners[i].first != -1)
			{
				if (q19.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q19.centre.y)
					{
						quads[n].number = 24;
						indexQ24 = n;
					}
				}
			}
			if (q23.associatedCorners[i].first != -1)
			{
				if (q23.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q23.centre.y)
					{
						quads[n].number = 27;
						indexQ27 = n;
					}
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
	Quad& q24 = quads[indexQ24];
	Quad& q27 = quads[indexQ27];
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
			quadsInRow.push_back(&q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i]->number = currentQuadIndex;
		currentQuadIndex++;
	}
	q27.number = 27;
	currentQuadIndex = 28;

	// Number the two attached to q1 and q5
	int indexQ28 = -1;
	int indexQ32 = -1;
	for (int n = 0; n < quads.size(); ++n)
	{
		for (int i = 0; i < 4; ++i)
		{
			if (q24.associatedCorners[i].first != -1)
			{
				if (q24.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q24.centre.y && quads[n].centre.x < q24.centre.x)
					{
						quads[n].number = 28;
						indexQ28 = n;
					}
				}
			}
			if (q27.associatedCorners[i].first != -1)
			{
				if (q27.associatedCorners[i].first == quads[n].id)
				{
					if (quads[n].centre.y > q27.centre.y && quads[n].centre.x > q27.centre.x)
					{
						quads[n].number = 32;
						indexQ32 = n;
					}
				}
			}
		}
	}
	if (indexQ28 == -1 || indexQ32 == -1)
	{
		// we failed? Return. SHould have an error code. At least false?
		return;
	}

	// Final row
	Quad& q28 = quads[indexQ28];
	Quad& q32 = quads[indexQ32];
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
			quadsInRow.push_back(&q);
		}
	}
	sort(quadsInRow.begin(), quadsInRow.end(), CompareQuadByCentreX);

	// Number
	for (int i = 0; i < quadsInRow.size(); ++i)
	{
		quadsInRow[i]->number = currentQuadIndex;
		currentQuadIndex++;
	}
	q32.number = 32;

	// Renormalise all quad points
	for (Quad& q : quads)
	{
		q.centre.x /= gtSize.x;
		q.centre.y /= gtSize.y;
	}
}

/*
	Get the intrinsic and extrinsic parameters from the homography

	THIS DOES NOT QORK NEED QR DECOMPOSITIOn

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

/*
	Given a series of correspondences between images (in normalised coords),
	compute a pinhole model calibration, if possible, and return it

	TODO - explain how Zhang does it
	name matrix S, not V

	This homography is not the solution to it all, cos different equations have different rotations
	etc but it encodes the calibration for the camera. We use the equations in Zhang, 1992, to solve
	for the parameters of K. Then build the matrix, and return it. 

	This can fail when the matrix S is rank deficient; that is, oversolved, such that rows conflict and
	there are infinitely many solutions
*/
bool ComputeCalibration(const std::vector<Matrix3f>& estimates, Matrix3f& K)
{
	// Construct the system of linear equations in the parameters of B
	// The size of V is 2x total quads by 6
	MatrixXf V;
	V.resize(estimates.size() * 2, 6);
	V.setZero();
	for (unsigned int i = 0; i < estimates.size(); ++i)
	{
		// Compute the vectors 
		const Matrix3f& H = estimates[i];
		// These are 6-vectors, but eigen doesn
		VectorXf v11(6);
		VectorXf v12(6);
		VectorXf v22(6);

		// i = j = 1
		v11(0) = H(0, 0)*H(0, 0);
		v11(1) = H(0, 0)*H(0, 1) + H(0, 1)*H(0, 0);
		v11(2) = H(0, 1)*H(0, 1);
		v11(3) = H(0, 2)*H(0, 0) + H(0, 0)*H(0, 2);
		v11(4) = H(0, 2)*H(0, 1) + H(0, 1)*H(0, 2);
		v11(5) = H(0, 2)*H(0, 2);

		// i = 1. j = 2
		v12(0) = H(0, 0)*H(1, 0);
		v12(1) = H(0, 0)*H(1, 1) + H(0, 1)*H(1, 0);
		v12(2) = H(0, 1)*H(1, 1);
		v12(3) = H(0, 2)*H(1, 0) + H(0, 0)*H(1, 2);
		v12(4) = H(0, 2)*H(1, 1) + H(0, 1)*H(1, 2);
		v12(5) = H(0, 2)*H(1, 2);

		// i = j = 2
		v22(0) = H(1, 0)*H(1, 0);
		v22(1) = H(1, 0)*H(1, 1) + H(1, 1)*H(1, 0);
		v22(2) = H(1, 1)*H(1, 1);
		v22(3) = H(1, 2)*H(1, 0) + H(1, 0)*H(1, 2);
		v22(4) = H(1, 2)*H(1, 1) + H(1, 1)*H(1, 2);
		v22(5) = H(1, 2)*H(1, 2);

		// Use these to create the matrix V
		// [  v_12 transpose         ] 
		// [  (v_11 - v_22) transpose] b = 0
		// as per zhang, so form these equations in V
		V(2 * i, 0) = v12(0);
		V(2 * i, 1) = v12(1);
		V(2 * i, 2) = v12(2);
		V(2 * i, 3) = v12(3);
		V(2 * i, 4) = v12(4);
		V(2 * i, 5) = v12(5);

		V(2 * i + 1, 0) = v11(0) - v22(0);
		V(2 * i + 1, 1) = v11(1) - v22(1);
		V(2 * i + 1, 2) = v11(2) - v22(2);
		V(2 * i + 1, 3) = v11(3) - v22(3);
		V(2 * i + 1, 4) = v11(4) - v22(4);
		V(2 * i + 1, 5) = v11(5) - v22(5);
	}

	// Get the singular values from the decomposition
	BDCSVD<MatrixXf> svd(V, ComputeThinU | ComputeFullV);
	if (!svd.computeV())
		return false;
	auto& v = svd.matrixV();

	// We get a six-vector out of this. Get the singular values for our vector
	// see the code from work about this section. 
	// We use the matrix form of this vector (see Zhang) when pulling it out of V;
	// Set B to be the column of V corresponding to the smallest singular value
	// which is the last as singular values come well-ordered
	// B = (B11, B12, B22, B13, B23, B33)
	VectorXf B(6);
	B << v(0, 5), v(1, 5), v(2, 5),
		v(3, 5), v(4, 5), v(5, 5); // TODO - is this right?
	cout << B << endl;
	/*
	Now that we have the parameters of B, compute parameters of K. 
	This is using Zhang's equations from his Appendix B, 
	where B = lambda A-T A
	Since we don't care about scale, we divide throughout by lambda once we have it
	*/
	float principalY = (B(1)*B(3) - B(0)*B(4)) / (B(0)*B(2) - B(1)*B(1));
	float lambda = B(5) - (B(3)*B(3) + principalY*(B(1)*B(3) - B(0)*B(4)))/B(0);
	float focalX = sqrt(lambda/B(0));
	float focalY = sqrt(lambda*B(0)/(B(0)*B(2) - B(1)*B(1)));
	float skew = -B(1)*focalX*focalX*focalY/lambda;
	float principalX = (skew*principalY/focalY) - B(3)*focalX*focalX/lambda;
	
	// TODO:
	// SOmetimes getting NaN cos taking sqrt of negative
	// Also getting wildly different numbers
	// I think this is because the checker detection is not very consistent

	K.setZero();
	K(0, 0) = focalX;
	K(1, 1) = focalY;
	K(2, 2) = lambda;
	K(0, 1) = skew;
	K(0, 2) = principalX;
	K(1, 2) = principalY;
	//K /= lambda; // is this necessary?

	return true;
}