# Calibration

TODO: comment all the code appropriately

Feedback: This tutorial should be standalone

This exercise is to learn the basics of single camera calibration (can easily be extended to multiple views). As with my [last tutorial](https://github.com/dmckinnon/stitch), the best place to start is Main.cpp, where the components mentioned below are used in sequence. It's a little different in this one to panorama stitching, but I'vew tried to comment the code as best I can for maximum readability. It's not quite written in the same exercise-like format as panorama stitching, since there is less to tweak here and things are more prescribed than described. 

This tutorial follows a paper written in 1998 by Microsoft Research researcher Zhang - [Zhang Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) - which is a simplistic yet sufficient method of calibrating a single camera using a checkerboard. His initial results and data can be found [here](https://www.microsoft.com/en-us/research/project/a-flexible-new-technique-for-camera-calibration-2/?from=http%3A%2F%2Fresearch.microsoft.com%2F~zhang%2Fcalib%2F). For another well-detailed, pseudo-coded explanation of Zhang, but with the mathematics fleshed out, see [this well-written paper by Burger](http://staff.fh-hagenberg.at/burger/publications/reports/2016Calibration/Burger-CameraCalibration-20160516.pdf).

# Contents:
1. Overview of the README and Calibration
2. Checkerboard Recognition
3. Initial Parameter Estimation
4. Refinement
5. Reading material


## Overview

For those of you who prefer wikipedia, perhaps start [here](https://en.wikipedia.org/wiki/Camera_resectioning). Everyone else, read on. 

Camera calibration is a process for a camera that will give its intrinsic parameters (and in some cases extrinsic). What are these? The [camera intrinsics](http://ksimek.github.io/2013/08/13/intrinsic/) are the values for focal length, skew and principal point. The extrinsic parameters are simply the parameters to place the camera in a particular point in space relative to another point - for a single camera this makes little sense, but for a multi-camera rig, you'd want to know the position of each camera relative to the others, and that's what this would give. The other thing that calibration would give you is, assuming a good model, parameters for your [lens distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) model.

The basic algorithm is that we have a synthetic image of a checkerboard, checkerboard.jpg, and some captured images of a printed version of this - these are 1, 2, 3 .jpg. First, try and figure out how each photo lines up with the synthetic image, via a transform called a homography. The equation is, if m is an image point and M is the corresponding point on the synthetic checkerboard, *s*m = HM, where *s* is some scale factor. H is a 3x3 matrix that transform M = (x,y,1) (we act as if this is in 3D space, at Z = 1) to some (1/w)(x', y', 1), which is the 3D (or projective) coordinates for the photo. H can be split into two other matrices - H = K * P. Here, P is what's called a rigid-body transform - that is, a rotation and a translation. If we have some vector x, think of Px as Rx + t, where R is a rotation and t a translation. It rotates a vector, then shifts it. This just moves the point from the synthetic plane to the captured plane. K is more interesting. K is our camera matrix. You can read more detail below, but essentially this takes intp account focal length, centre offset, image skew, etc, and moves the point from general 3D space into camera space. 

The point of all this is that the transform we get from the photo of the checkerboard to the synthetic checkerboard captures these camera parameters. We then use this transform, from a bunch of images to get as much data as we can on the parameters, to guess at the camera matrix, and then refine our guess. 

### The code and components
Before I get into the nitty-gritty of the camera parameters, for those gluttons of theory, let's go over the higher level view of what's happening here. The input to this whole process is a synthetic image of a checkerboard, which is given here in checkerboard.jpg, and at least three images of that checkerboard from the camera, including multiple angles and distances. 
The output is the camera parameters. 

In essence this process involves three major steps: recognising the checkerboard, building an initial estimation of the parameters, and then refining the parameters. Each of these will be detailed here, or linked to explanations elsewhere. 

As my style goes, my main is laid out roughly in this order. I detect the checkers in the synthetic image first, and then in each captured image - this is the loop in main. Then an initial estimate of the camera parameters is found. Finally, the refinement function is called. These blocks are on a high level easy to follow, and each can be dived into for more detail. Calibration.cpp and Estimation.cpp contain most of the big-picture code, with Image.cpp containing smaller level operations.

#### Intrinsics in more detail
If you didn't read the [intrinsics link](http://ksimek.github.io/2013/08/13/intrinsic/) above, i'll go into more detail here. The following discussion assumes a [pinhole model](https://en.wikipedia.org/wiki/Pinhole_camera_model) of a camera. The focal length of a camera is the length between the lens hole and the actual sensor. You can imagine that if you move the sensor further from the lens hole, then it will see a smaller and smaller image, and the closer it gets the wider an image you see - this is because we change where the sensor is relative to the point of focus of the length (which for a pinhole can be anywhere but realistically isn't, there's usually one spot that can focus well). In theory, the focal length in x and in y are equal; in practice, they rarely are, which leads to the *aspect ratio* - the ratio between the x and y focal lengths. 

The principal point is the point on the film that lies on the principal axis, which is the line perpendicular to the image plane and passes through the pinhole lens. The principal point offset, captured in a camera's intrinsics, is the location of the principal point relative to the film's origin in this camera's case. Basically, the sensor may not be perfectly centred with respect to the lens, and this gives an offset, that we hope to measure and capture. Finally, there is skew. Skew is a parameter that might cause an image to lean to one side or the other - also called a [shear distortion](https://en.wikipedia.org/wiki/Shear_mapping).

These basic elements are what is captured in the [camera matrix](https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters) - an upper triangular 3x3 matrix that we use to project pixels into the real world and vice-versa. If you haven't looked this up, I highly recommend it at this stage. This is also in the links above. 

Another parameter set we might want to capture, depending on the problem, is lens distortion. This is where ... well, the lens ... distorts the image somehow. [There are a lot of different types of distortion](https://photographylife.com/what-is-distortion), depending on the lenses you use. If you're trying to accurately project points from the image to the world or vice-versa, then you need to be able to model this accurately. 



## Checkerboard Recognition

This step occurs on a per-image basis. This again can be broken down, into two steps:
- Finding the checkers and corners

- associating the checkers correctly between the captured image and the synthetic image. 

For this process, I'm going to be following [Scarramuzza's example](http://rpg.ifi.uzh.ch/docs/IROS08_scaramuzza_b.pdf). At first, I tried to use Harris corner detection, which found some corners, but then I couldn't figure out how to associate those corners. This method is significantly slower, but is far more accurate and robust than anything I could think of. Thus, I shall stand on the shoulders of giants ... or so I thought. I freely admit that while my method follows the general theory well, it has a few smaller bugs that I couldn't quite iron out. For the most part it works. 

So Scarramuzza's paper builds on top of the checker detection done by [OpenCV](https://docs.opencv.org/2.4.13.7/doc/tutorials/calib3d/camera_calibration/camera_calibration.html). I optimised my operations by deliberately capturing images with only a white background, to make things easier. First, threshold the image. There are [many different methods](https://en.wikipedia.org/wiki/Thresholding_(image_processing)) of doing this, that should all achieve the same basic result - the image is either white or black, with hopefully the checkers black and everything else white. Then, to detect the checkers, I want to look for edges. My method gathers blobs by doing a [flood fill](https://www.techiedelight.com/flood-fill-algorithm/) on black squares, noting all those black squares that touch a white square (that is, the edges), and then using [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) to find four lines amongst these edge pixels. This finds the four edges of each checker. For each checker, the intersections of these lines gives the corners, and the average of all points in the checker gives the centre. 

Enter Scarramuzza's ideas. So far this has been a sufficient analog of what OpenCV was doing. The next step is to match the corners of each checker - meaning corner X of checker Y touches corner Z of checker W, and so on. This is useful for defining which checkers are the edges and corners (those with two or one connecting checkers), which comes into play during the recognition step. Scarramuzza gets pairs of checkers and finds the line through the centre perpendicular to each side. If you draw this, you'll see that the intersection of these two lines forms a quadrilateral. If a corner from each checker lies in this quad, and the centres are close enough, the corners match. If you need more clarity, or perhaps pictures, take a look at his paper - linked above. 

So by this stage we hopefully have detected every checker, and we know how all the checkers are arranged with respect to each other. We also know which checkers are the ones on the corners. This gives us four known points corresponding to the corner checkers in the synthetic image, which are easy to find (just find the top left and right, and bottom left and right checkers). Given these correspondences, we use [SVD](https://github.com/dmckinnon/stitch#finding-the-best-transform) to find the best homography between the captured checkerboard and the synthetic checkerboard. If we find one that works, we can then transform the captured checkerboard into the plane of the synthetic checkerboard, where everything is nicely arranged in easily identifiable rows. 

There may be better methods of registration than this; in fact, I'm sure there is. But this works. It's not nice, but it works. If I was doing this again, I'd scrap so much that this would be irrelevant, but whatever. 
I just go along the rows from top to bottom and number the checkers 1 to 32 as intended. Simple, easy. 

So now we have a captured set of checkers and we have numbered them in correspondence with all the synthetic checkers. And this is done across all captured images. We have a homography for each image. Now we can take an initial guess at the camera calibration matrix. 


## Initial Parameter Estimation

This is fairly simple in some ways, and complex in others. Once we have the correct association of checkers between the captured image and synthetic image, we can use those pairs of matching points to form a homography between the planes. For a detailed description of this process that I've already written and don't care to repeat, see [here](https://github.com/dmckinnon/stitch#finding-the-best-transform). This is done with SVD, which is explained in that link (There's a lot and I see little point in copying and pasting). 
But then comes the not-obvious part, and I need to refer to Zhang. In section 2.3, Zhang describes several constraints on the camera matrix, given a homography between the image and the synthetic checkers. Then in 3.1, he goes over a method to turn this into a system of linear equations for multiple homographies, such that solving these equations via SVD or some other method will yield the camera parameters (See Zhang, Appendix B). It's hard and it took me some working through to understand. If you don't understand it ... that's perfectly ok. If you're trying to implement it ... well, so long as you can type the math up correctly, that's what matters. 

This system of linear equations yields a set of camera parameters that provide an initial linear-least-squares guess to fit all the homographies. Next, we refine, using all the centres of the checkers we detected. 

## Refinement

[Under this description of how to find transforms](https://github.com/dmckinnon/stitch#finding-the-best-transform), you'll see a sub-heading called Optimisation. That describes the same process we use here. It's not exactly the same, as we are optimising a mathematically different function, but the concepts are the same. 

So the parameters we are optimising over are the intrinsic camera parameters - focal length in x and y, skew, and the principal point in x and in y - and the extrinsic parameters for each image; that is, the rotation and translation from the camera for that image to the 'camera' for the synthetic image. I use a six-vector for these - three rotation parameters and three translation parameters - and use an element of the [Special Euclidean group](http://planning.cs.uiuc.edu/node147.html) to represent and enact the rotation and translation. Finally, there are the camera distortion parameters, for which Zhang has two, k_0 and k_1 (see Zhang section 3.3 - these are polynomial coefficients). 
This leads to one massive equation that we are trying to minimise. It boils down to:
For each image
    For each checker in the image
        total_error += synthetic checker - lens_distortion * camera matrix * rotation and translation * captured checker
        
We then get the jacobian of this monstrous function with respect to all the parameters, and as I explain better in my [other writing on optimisation](https://github.com/dmckinnon/stitch#finding-the-best-transform) we use this in the Levenberg-Marquardt algorithm to minimise this error.  

## Other notes
I freely admit that as a whole this doesn't function perfectly. Yes, mostly it runs, and it completes and prints out a camera matrix. I don't trust this camera matrix, and there are some weird bugs, like sometimes the homography fails on checker sets it has previously succeeded on. I think this is a checker detection bug. However, the theory is correct. I've checked through it all. So you can rest assured on that. And it should give a sufficient start if you want to try this on your own. 

## Building and running
I've included in the repo the .exe and the necessary dlls to just run this straight out of the box. It takes in two command-line arguments - the folder where the images are, and the number of captured images, and prints out the camera matrix to the command line. If something fails, it prints error messages.
Eg. 
> calibration.exe C:\Users\fakeuser\Pictures\checkerboard_pics\ 3

I developed this in Visual Studio on Windows, but nothing is platform-dependent. The only dependencies it has are Eigen and OpenCV, just for a few things like the Mat types, Gaussian blur, etc. If you don't want to do the whole download, build OpenCV yourself thing, just use the dlls and libs I supplied, and download the source code and be sure to link the headers the right way. Installing OpenCV was complicated enough for me that this is a topic for another day, and unfortunately I can't find the link yet that I used. If I find it, I'll add it.

Thanks for reading - enjoy!


## Links for further reading:

Tips for how to best take photos for calibration: https://pgaleone.eu/computer-vision/2018/03/04/camera-calibration-guidelines/

An explanation of Levenberg-Marquardt, and of Zhang: https://staff.fnwi.uva.nl/l.dorst/hz/hz_app6_zhang.pdf



