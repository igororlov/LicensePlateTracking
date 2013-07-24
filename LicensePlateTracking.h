#ifndef LICENSEPLATETRACKING_H
#define LICENSEPLATETRACKING_H

// std includes
#include <iostream>
using namespace std;

// opencv includes
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\stitching\stitcher.hpp>
using namespace cv;

// ipp includes
#include <ippcore.h>
#include <ippi.h>
#include <ippm.h>


void onMouseCallback(int event, int x, int y, int , void* );

IplImage* ippResize(Mat& src, double xFactor, double yFactor);

void detectKeypoints(Mat& image, int fastThreshold, Rect roiRect, vector<KeyPoint>& keypoints);

void trackKeypoints(Mat& currentFrame, Mat& previousFrame, vector<KeyPoint>& currentKeypoints, vector<KeyPoint>& previousKeypoints);

#endif