#include "LicensePlateTracking.h"

#define FILENAME "C:\\Main\\Work\\Video\\kazahstan.avi"
#define COLOR_PURPLE Scalar(255,0,255)

// предположение о максимально возможном смещении номера на след. кадре
#define maxHorisontalShift 100 
#define maxVerticalShift 80

static Point topLeft(0, 0);
static Point botRight;
static bool isSettingTopLeft = true;

int main() {
	VideoCapture capture;
	Mat currentFrame, previousFrame;
	int nFrames, frameNo;
	vector<KeyPoint> currentKeypoints, previousKeypoints;
	bool stop = false;
	double rate, delay;

	capture.open(FILENAME);
	nFrames = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
	rate = capture.get(CV_CAP_PROP_FPS);
	delay = 1000 / rate;
	botRight = Point((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));

	namedWindow("Frame");
	setMouseCallback("Frame", onMouseCallback, 0);

	frameNo = 0;

	while ( !stop ) {
		previousFrame.data = currentFrame.data;
		swap(currentKeypoints, previousKeypoints);
		currentKeypoints.clear();
		frameNo++;

		if ( !capture.read(currentFrame) ) {
			break;
		}
		
		IplImage* resizedIpl = ippResize(currentFrame, 1, 1);
		Mat resized(resizedIpl);
		
		Rect roiRect(topLeft, botRight);
		detectKeypoints(resized, 25, roiRect, currentKeypoints);
		//drawKeypoints(resized, currentKeypoints, resized, COLOR_PURPLE); // времяемкая!
		printf("Frame: %d; currentFrame Address: %u; resized Address: %u;\n", frameNo, currentFrame.data, resized.data);
		
		int key = waitKey((int)delay); 
		if ( key == 27 ) {
			stop = true;
		}
		else if ( key == 32 ) {	
			waitKey(0);
		}

		imshow("Frame", resized);

		previousFrame.release();		
		cvReleaseData(resizedIpl);
	}

	capture.release();

	return 0;
}

void onMouseCallback(int event, int x, int y, int flags, void* param) {
	if ( event == CV_EVENT_LBUTTONDOWN ) {
		if ( isSettingTopLeft ) {
			if ( x <= botRight.x && y <= botRight.y ) {
				topLeft.x = x;
				topLeft.y = y;
			}
		} else {
			if ( x >= topLeft.x && y >= topLeft.y ) {
				botRight.x = x;
				botRight.y = y;
			}
		}
		isSettingTopLeft = !isSettingTopLeft;
	}
}

IplImage* ippResize(Mat& src, double xFactor, double yFactor) {
	IplImage *iplImage = cvCreateImageHeader(cvSize(src.cols, src.rows), IPL_DEPTH_8U, 1);
	iplImage->imageData = (char*)src.data;
	IppiSize srcSize = { src.cols, src.rows };
	int srcStep = src.step;
	IppiRect srcRoi = { 0, 0, src.cols, src.rows };
	int channels = src.channels();

	IplImage *pTarget = cvCreateImage(cvSize(src.cols * xFactor, src.rows * yFactor), IPL_DEPTH_8U, channels);
	int dstStep = pTarget->widthStep;
	IppiSize dstRoiSize = { pTarget->width, pTarget->height };

	if ( channels == 1 ) {
		IppStatus status = ippiResize_8u_C1R((const Ipp8u*)iplImage->imageData, srcSize, srcStep, srcRoi, 
			(Ipp8u*)pTarget->imageData, dstStep, dstRoiSize, xFactor, yFactor, IPPI_INTER_LINEAR);
	} else {
		IppStatus status = ippiResize_8u_C3R((const Ipp8u*)iplImage->imageData, srcSize, srcStep, srcRoi, 
			(Ipp8u*)pTarget->imageData, dstStep, dstRoiSize, xFactor, yFactor, IPPI_INTER_LINEAR);
	}

	return pTarget;
}

void detectKeypoints(Mat& image, int fastThreshold, Rect roiRect, vector<KeyPoint>& keypoints) {
	FastFeatureDetector fastDetector(fastThreshold);
	Mat roiImg;
	roiImg = image(roiRect);

	fastDetector.detect(roiImg, keypoints);

	for ( int i = 0; i < keypoints.size(); i++ ) {
		keypoints[i].pt.x += roiRect.x;
		keypoints[i].pt.y += roiRect.y;
	}
}