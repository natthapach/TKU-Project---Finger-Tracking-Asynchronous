#pragma once
#include "opencv2/opencv.hpp"
#include <OpenNI.h>
#include <NiTE.h>
#include <thread> 

using namespace std;

class KinectReader {
public :
	int initialize();
	void readDepthFrame();
	void readRGBFrame();
	cv::Mat getDepthFrame();
	cv::Mat getRGBFrame();
	cv::Mat getDepthHandMask();
	cv::Mat getRawDepthFrame();

	bool isHandTracking();
	float getHandPosX();
	float getHandPosY();
	int getHandDepth();
	int getDepthHandRange();
	cv::Point getHandPoint();
	int getHandRadius(int mm);

	void convertDepthToColor(int x, int y, int z, int *cx, int *cy);
	void convertDepthToWorld(float x, float y, float z, float *wx, float *wy, float *wz);
protected :
	const int RANGE = 100;
	openni::Device device;
	openni::VideoStream colorStream;
	openni::VideoStream depthStream;
	nite::HandTracker handTracker;
	nite::HandTrackerFrameRef handsFrame;
	int depthHistogram[65536];
	uint16_t depthRaw[480][640];
	uchar img[480][640][3];
	uchar mask[480][640];
	uchar maskColor[480][640];

	cv::Mat colorFrame = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat colorDepthFrame;
	cv::Mat depthFrame;
	cv::Mat depthHandMask;

	float handPosX = 0;
	float handPosY = 0;
	int handDepth = 0;
	int numberOfHands = 0;
private :
	thread readRGBThread;
	thread readDepthThread;

	void asyncReadRGBFrame();
	void asyncReadDepthFrame();

	void calDepthHistogram(openni::VideoFrameRef depthFrame, int * numberOfPoints, int * numberOfHandPoints);
	void modifyImage(openni::VideoFrameRef depthFrame, int numberOfPoints, int numberOfHandPoints);
	void settingHandValue();
};