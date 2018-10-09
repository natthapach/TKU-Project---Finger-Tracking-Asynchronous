#pragma once
#include "KinectReader.h"

class Application {
public :
	int initialize();
	void start();
protected :
	ushort maxDepth = 65535;
	ushort minDepth = 0;

	const int DISTANCE_THESHOLD = 10;
	int64 tickCount = 0;
	long frameCount = 0;
	time_t startTimestamp;
	double estimateFPS;

	cv::Mat skinMask;
	cv::Mat colorFrame;
	cv::Mat depthFrame;
	cv::Mat rawDepthFrame;
	cv::Mat edgeColorFrame;
	cv::Mat histogramFrame;
	cv::Mat handMask;
	cv::Mat handLayer1 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer2 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer3 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);

	vector<cv::Point> handLayer1Corners;
	vector<vector<cv::Point>> contoursL1;
	map<int, vector<cv::Point>> cornerGroup;
	vector<cv::Point> fingerL2Point;
	vector<cv::Point> hullL2;

	cv::MatND hist;
	

	void transformColorFrame();
	void buildEdgeColor();
	void buildSkinMask();
	void buildDepthHandMask();
	void combineSkinHandMask();
	void buildHand3Layers();
	void buildHistogram();

	void evaluateHandLayer1();
	void evaluateHandLayer2();
	void evaluateLayer12();

	void clusterPoint(vector<cv::Point>& inputArray, vector<cv::Point>& outputArray);
private:
	KinectReader kinectReader;
	void calculateContourArea(vector<cv::Point> contour, double *area);
};