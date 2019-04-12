# Calibration

This exercise is to learn the basics of single camera calibration (can easily be extended to multiple views). As with my [last tutorial](https://github.com/dmckinnon/stitch), the best place to start is Main.cpp, where the components mentioned below are used in sequence. It's a little different in this one to panorama stitching, but I'vew tried to comment the code as best I can for maximum readability. This tutorial follows a paper written in 1998 by Microsoft Research researcher Zhang - [Zhang Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) - which is a simplistic yet sufficient method of calibrating a single camera using a checkerboard. 

# Contents:
1. Overview of the README and Calibration
2. Checkerboard Recognition
3. Initial Parameter Estimation
4. Refinement
5. Reading material


## Overview

For those of you who prefer wikipedia, perhaps start [here](https://en.wikipedia.org/wiki/Camera_resectioning). Everyone else, read on. 

Camera calibration is a process for a camera that will give its intrinsic parameters (and in some cases extrinsic). What are these? The [camera intrinsics](http://ksimek.github.io/2013/08/13/intrinsic/) are the values for focal length, skew and principal point. The extrinsic parameters are simply the parameters to place the camera in a particular point in space relative to another point - for a single camera this makes little sense, but for a multi-camera rig, you'd want to know the position of each camera relative to the others, and that's what this would give. THe other thing that calibration would give you is, assuming a good model, parameters for your [lens distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) model.

### The code and components
Before I get into the nitty-gritty of the camera parameters, for those gluttons of theory, let's go over the higher level view of what's happening here. The input to this whole process is a synthetic image of a checkerboard, which is given here in checkerboard.jpg, and at least three images of that checkerboard from the camera, including multiple angles and distances. 
The output is the camera parameters. 

In essence this process involves three major steps: recognising the checkerboard, building an initial estimation of the parameters, and then refining the parameters. Each of these will be detailed here, or linked to explanations elsewhere. 

#### Intrinsics in more detail
If you didn't read the [intrinsics link](http://ksimek.github.io/2013/08/13/intrinsic/) above, i'll go into more detail here. The following discussion assumes a [pinhole model](https://en.wikipedia.org/wiki/Pinhole_camera_model) of a camera. The focal length of a camera is the length between the lens hole and the actual sensor. You can image that if you move the sensor further from the lens hole, then it will see a smaller and smaller image, and the closer it gets the wider an image you see - this is because we change where the sensor is relative to the point of focus of the length (which for a pinhole can be anywhere but realistically isn't, there's usually one spot that can focus well). In theory, the focal length in x and in y are equal; in practice, they rarely are, which leads to the *aspect ratio* - the ratio between the x and y focal lengths. 

The principal point is the point on the film that lies on the principal axis, which is the line perpendicular to the image plane and passes through the pinhole lens. The principal point offset, captured in a camera's intrinsics, is the location of the principal point relative to the film's origin in this camera's case. Basically, the sensor may not be perfectly centred with respect to the lens, and this gives an offset, that we hope to measure and capture. Finally, there is skew. Skew is a parameter that might cause an image to lean to one side or the other - also called a [shear distortion](https://en.wikipedia.org/wiki/Shear_mapping).

These basic elements are what is captured in the [camera matrix](https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters) - an upper triangular 3x3 matrix that we use to project pixels into the real world and vice-versa. If you haven't looked this up, I highly recommend it at this stage. It's also in the links above. 

Another parameter set we might want to capture, depending on the problem, is lens distortion. This is where ... well, the lens ... distorts the image somehow. [There are a lot of different types of distortion](https://photographylife.com/what-is-distortion), depending on the lenses you use. If you're trying to accurately project points from the image to the world or vice-versa, then you need to be able to model this accurately. 

###



## Checkerboard Recognition

This step occurs on a per-image basis. This again can be broken down, into two steps:
- Finding the checkers and corners

- associating the checkers correctly between the captured image and the synthetic image. 

For this process, I'm going to be following [Scarramuzza's example](http://rpg.ifi.uzh.ch/docs/IROS08_scaramuzza_b.pdf). At first, I tried to use Harris corner detection, which found some corners, but then I couldn't figure out how to associate those corners. This method is significantly slower, but is far more accurate and robust than anything I could think of. Thus, I shall stand on the shoulders of giants. 


## Initial Parameter Estimation

This is fairly simple in some ways, and complex in others. Once we have the correct association of checkers between the captured image and synthetic image, we can use those pairs of matching points to form a homography between the planes. For a detailed description of this process that I've already written and don't care to repeat, see [here](https://github.com/dmckinnon/stitch#finding-the-best-transform). This is done with SVD, which is explained in that link (There's a lot and I see little point in copying and pasting). 
<some stuff about distortion?>

## Refinement

Back on my panorama stitching tutorial, under the heading [Finding the best transform](https://github.com/dmckinnon/stitch#finding-the-best-transform), you'll see a sub-heading called OPtimisation. That describes the same process we use here. It's not exactly the same, as we are optimising a mathematically different function, but the concepts are the same. 



## Links for reading:

tips for image cal: https://pgaleone.eu/computer-vision/2018/03/04/camera-calibration-guidelines/


http://staff.fh-hagenberg.at/burger/publications/reports/2016Calibration/Burger-CameraCalibration-20160516.pdf


http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.534&rep=rep1&type=pdf

https://staff.fnwi.uva.nl/l.dorst/hz/hz_app6_zhang.pdf



---- Checkerboard detection

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.501.6514&rep=rep1&type=pdf


http://www.cvlibs.net/software/libcbdetect/


Scarramuzza - http://rpg.ifi.uzh.ch/docs/IROS08_scaramuzza_b.pdf

Automatic Chessboard Detection for Intrinsic and Extrinsic
Camera Parameter Calibration 


