# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test\_video.mp4 and later implement on full project\_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

# Classification
Color transformations and histograms of colors have been used to extract features for cars, which are then concatenated with the HOG features. The following list shows the values of the variables that were chosen/tuned:
1. colorspace='YCrCb'
2. orient=9
3. pix\_per\_cell=8
4. cell_\per\_block=2
5. hist_bins=16
(TODO: Test effect of hog\_channel, spatial\_size and hist\_range)
An SVM was then trained on this set resulting in a test accuracy of around 0.986, which seems a tad high.
REF:https://github.com/CYHSM/carnd/tree/master/CarND-Vehicle-Detection

# Methodology
The code from the lecture slides was used to implement a sliding window approach. The heat map was produced by adding 1 to the areas within the bounding boxes. These areas were then depicted using rectangles. 

# Pipeline
A function called process\_video() calls process\_frame() which processes each frame. A smoothing of 20 was used to reduce false positives as a car is continuously detected in consecutive frames. As mentioned in the discussion this methodology need improvement.

# Discussion
Given the very high test accuracy, there might exist a bias towards the training data. The TODOs that still remain will help improve accuracy. The code definitely needs optimisation in perms of parallelising as it is still rather slow. The process frame function might be called on multiple frames simultaneously to help with that, but given time constraints it will have to wait for after the end of term. A function to clean out the rectangles around the bounding boxes to ensure that the car is correctly enclosed only by a single outline is currently in the works, but still needs some work, and once again, given time constraints it will have to wait. An addition of tracking the cars detected through frames rather than constantly processing each frame would help speed up the process too.

Please not that the file single_run.py was used to generate the images required by the rubric, and vehdet.py to process the entire video.

## Changes after initial feedback:

#Addition of multiple scales
1. The code has been modified to ensure that the white car is not missed in the time frame mentioned above. Scaling has been added to deal tith the issue. The scaling now includes 1.3 and 2.1 which leads to better accuracy. Going beyond 2.1 does not improve results while going below 1.3 leads to a significantly larger number of false positives at some distance in front of the car. 

#Discussion on how the hyper paratemeters were chosen/tuned; as well as use of heatmap to deal with false positives/multiple detections
2. As for the methodology of arriving at these values it was using trial and error. The color sapce chosen is more robust to brightness variations The other values were played arbitrarily selected, with some help for a starting point based on th electure videos, but these seem to provide the best results, as modifying these independantly leads to a detrement in the final identification. While reducing the overlap does lead to some speed up, it is not worth it at the cost of final result. A heatmap has been used, which addresses the false positives issue to a large extent. That said, adding multiple scale values to allow for detection of far away vehicles does introduce more false positives. A better training set which simply shows pieces of road to a larger extent as non_cars would definitely help.

The video without using the multiple scaling factors (project\_out\_old.mp4) has been left in the submission along with the new video (project\_out\_new.mp4). It was tuned with the following reasoning in mind: 
1. it has the fewest false positives that would lead to the car constantly accelerating and decelerating
2. the cars that have not been tagged are far away and 2 lanes away, which might be better suited for autonomous driving
