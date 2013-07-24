#include "LicensePlateTracking.h"

#define FILENAME "C:\\Main\\Work\\Video\\kazahstan.avi"
//#define FILENAME "C:\\Main\\Work\\Video\\TWN.avi"
#define COLOR_PURPLE Scalar(255,0,255)
#define COLOR_GREEN Scalar(0,255,0)

// max possible plate coordinates shift on next frame
#define MAX_HORISONTAL_SHIFT 60 
#define MAX_VERTICAL_SHIFT 35

#define SCALE_X 0.5
#define SCALE_Y 0.5

#define FAST_THRESH 20

static Point2d topLeft(0, 0); // plate coords and size
static Point2d botRight;
static Size plateSize;

static bool isSettingTopLeft = true;
static bool isPlateRectangleDefined = false;
static bool isNewPlateDefined = true;

static vector<Point> plateCoords; 

int main() {
	VideoCapture capture;
	Mat currentFrame, previousFrame;
	int nFrames, frameNo, frameWidth, frameHeight;
	vector<KeyPoint> currentKeypoints, previousKeypoints;
	bool stop = false;
	double rate, delay;

	capture.open(FILENAME);
	nFrames = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
	rate = capture.get(CV_CAP_PROP_FPS);
	delay = 1000 / rate;
	frameWidth = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	frameHeight = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	botRight = Point(frameWidth, frameHeight);

	namedWindow("Frame");
	setMouseCallback("Frame", onMouseCallback, 0);

	frameNo = 0;

	while ( !stop ) {
		currentFrame.copyTo(previousFrame);
		currentFrame.release();
		swap(currentKeypoints, previousKeypoints);
		currentKeypoints.clear();
		frameNo++;

		if ( !capture.read(currentFrame) ) {
			break;
		}
		
		IplImage* resizedIpl = ippResize(currentFrame, SCALE_X, SCALE_Y);
		Mat resized(resizedIpl);
		currentFrame.release();
		resized.copyTo(currentFrame);

		if ( isPlateRectangleDefined ) {
			Point p1 = Point(topLeft.x - SCALE_X * MAX_HORISONTAL_SHIFT, topLeft.y - SCALE_Y * MAX_VERTICAL_SHIFT);
			Point p2 = Point(botRight.x + SCALE_X * MAX_HORISONTAL_SHIFT, botRight.y + SCALE_Y * MAX_VERTICAL_SHIFT);
			Rect roiRect(p1, p2);

			if ( p1.x > 0 && p1.y > 0 && p2.x < currentFrame.cols && p2.y < currentFrame.rows ) {
				detectKeypoints(currentFrame, FAST_THRESH, roiRect, currentKeypoints);
				//drawKeypoints(currentFrame, currentKeypoints, currentFrame, COLOR_PURPLE); // time-consuming!
				if ( isNewPlateDefined ) {
					previousKeypoints.clear();
					isNewPlateDefined = false;
				}
				if ( previousKeypoints.size() > 0 && currentKeypoints.size() > 0 ) {	
					trackKeypoints(currentFrame, previousFrame, currentKeypoints, previousKeypoints);
				}
			} else {
				isPlateRectangleDefined = false;
				plateCoords.clear();
			}
		}
		//printf("Frame: %d; currentFrame Address: %u; prevFrame Address: %u;\n", frameNo, currentFrame.data, previousFrame.data);
		//printf("Frame: %d; TopLeft(%d, %d); BotRight(%d, %d);\n", frameNo, topLeft.x, topLeft.y, botRight.x, botRight.y);

		int key = waitKey((int)delay); 
		if ( key == 27 ) {
			stop = true;
		}
		else if ( key == 32 ) {	
			waitKey(0);
		}
		
		imshow("Frame", currentFrame);
		previousFrame.release();		
		cvReleaseData(resizedIpl);
	}

	capture.release();

	return 0;
}

void onMouseCallback(int event, int x, int y, int flags, void* param) {
	if ( event == CV_EVENT_LBUTTONDOWN ) {
		if ( isSettingTopLeft ) {
			topLeft.x = x;
			topLeft.y = y;
			printf("TopLeft: (%d, %d);\n", x, y);
		} else {
			botRight.x = x;
			botRight.y = y;
			isPlateRectangleDefined = true;
			plateSize = Size(botRight.x - topLeft.x, botRight.y - topLeft.y);
			isNewPlateDefined = true;

			plateCoords.clear();
			plateCoords.push_back(Point((topLeft.x+botRight.x)/2,(topLeft.y+botRight.y)/2));
			printf("BotRight: (%d, %d); PlateSize: (%d, %d);\n", x, y, plateSize.width, plateSize.height);
		}
		isSettingTopLeft = !isSettingTopLeft;
	} else if ( event == CV_EVENT_RBUTTONDOWN ) {
		isPlateRectangleDefined = false;
		plateCoords.clear();
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

void trackKeypoints(Mat& currentFrame, Mat& previousFrame, vector<KeyPoint>& currentKeypoints, vector<KeyPoint>& previousKeypoints) {
	Mat currentFrameDescriptors, previousFrameDescriptors, matchesImage;
	vector<DMatch> matches;
	BruteForceMatcher<Hamming> matcher;
	BriefDescriptorExtractor extractor;
	
	extractor.compute(currentFrame, currentKeypoints, currentFrameDescriptors); // keypts size can change!
	extractor.compute(previousFrame, previousKeypoints, previousFrameDescriptors);

	if ( currentKeypoints.size() == 0 || previousKeypoints.size() == 0 ) {
		isPlateRectangleDefined = false;
		return;
	}

	try {
		matcher.match(previousFrameDescriptors, currentFrameDescriptors, matches);
	} catch (Exception ex) {
		cout << ex.msg << endl;
		cout << ex.err << endl;
	}

/*	drawMatches(previousFrame, previousKeypoints, currentFrame, currentKeypoints, matches, matchesImage);
	namedWindow("Matches");
	imshow("Matches", matchesImage);
	waitKey(0);*/

	int plateKeypointsCounter = 0;
	double xShift = 0.0, yShift = 0.0;

	for ( int i = 0; i < previousKeypoints.size(); i++ ) {
		Point kpt = previousKeypoints[i].pt;

		//if ( kpt.x >= topLeft.x && kpt.y >= topLeft.y && 
		//	kpt.x <= botRight.x && kpt.y <= botRight.y ) { // только те, что сейчас в номере

		KeyPoint currentKeypoint = currentKeypoints[matches[i].trainIdx];
		xShift += currentKeypoint.pt.x - kpt.x;
		yShift += currentKeypoint.pt.y - kpt.y;
		plateKeypointsCounter++;
		//}
	}
	xShift /= plateKeypointsCounter;
	yShift /= plateKeypointsCounter;

	//cout << "xShift: " << xShift << "; yShift: " << yShift << endl;

	topLeft.x += xShift;
	topLeft.y += yShift;
	botRight.x += xShift;
	botRight.y += yShift;

	Rect plateNewRect(topLeft, botRight);
	rectangle(currentFrame, plateNewRect, COLOR_GREEN, 2);
	
	plateCoords.push_back(Point(plateNewRect.x+plateNewRect.width/2, plateNewRect.y+plateNewRect.height/2));
	if ( plateCoords.size() > 1 )
	for ( int i = 1; i < plateCoords.size(); i++ ) {
		line(currentFrame, plateCoords[i], plateCoords[i-1], COLOR_GREEN);
	}

	matchesImage.release();
	currentFrameDescriptors.release();
	previousFrameDescriptors.release();
}

// TO DO:
void kalmanTracking() {
	KalmanFilter KF(4, 2, 0);

}
