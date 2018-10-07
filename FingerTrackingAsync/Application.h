#pragma once
#include "KinectReader.h"

class Application {
public :
	int initialize();
	void start();
protected :
	int64 tickCount = 0;
	long frameCount = 0;
	time_t startTimestamp;
	double estimateFPS;

	cv::Mat skinMask;
	cv::Mat colorFrame;
	cv::Mat depthFrame;
	cv::Mat rawDepthFrame;
	cv::Mat handMask;
	cv::Mat handLayer1 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer2 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer3 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);

	vector<cv::Point> handLayer1Corners;

	void transformColorFrame();
	void buildSkinMask();
	void buildDepthHandMask();
	void combineSkinHandMask();
	void buildHand3Layers();

	void evaluateHandLayer1();
private:
	KinectReader kinectReader;
	void calculateContourArea(vector<cv::Point> contour, double *area);
};