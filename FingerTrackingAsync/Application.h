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
	const double BUILD_LAYER_1_THRESHOLD = 0.2;
	const double BUILD_LAYER_2_THRESHOLD = 0.45;
	const double BUILD_LAYER_3_THRESHOLD = 0.6;

	const string WINDOW_RGB = "RGB";
	const string WINDOW_DEPTH = "Depth";
	const string WINDOW_MASK_L1 = "Mask Layer 1";
	const string WINDOW_MASK_L2 = "Mask Layer 2";
	const string WINDOW_MASK_L3 = "Mask Layer 3";
	const string WINDOW_HISTOGRAM = "Histogram";

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
	void buildRawHandHistogram();

	void evaluateHandLayer1();
	void evaluateHandLayer2();
	void evaluateHandLayer3();
	void evaluateLayer12();

	void clusterPoint(vector<cv::Point>& inputArray, vector<cv::Point>& outputArray, int thresh);

	void captureFrame();
private:
	KinectReader kinectReader;
	void calculateContourArea(vector<cv::Point> contour, double *area);
	int performKeyboardEvent(int key);
};