#pragma once
#include "KinectReader.h"
#define _USE_MATH_DEFINES
#include <cmath>
#define PI 3.14159265

class Application {
public :
	int initialize();
	void start();
protected :
	ushort maxDepth = 65535;
	ushort minDepth = 0;

	/* Percentage threshold for build hand layer 1 */
	const double BUILD_LAYER_1_THRESHOLD = 0.2;
	/* Percentage threshold for build hand layer 2 */
	const double BUILD_LAYER_2_THRESHOLD = 0.6;
	/* Percentage threshold for build hand layer 3 */
	const double BUILD_LAYER_3_THRESHOLD = 1.0;

	/* Threshold for select convexity defect on hand layer 2 */
	const int CONVEX_DEPTH_THRESHOLD_LAYER_2 = 10000;

	/* General threshold for cluster points */
	const int DISTANCE_THESHOLD = 10;
	/* Threshold for cluster corners on hand layer 1 */
	const int DISTANCE_THRESHOLD_CORNER_LAYER_1 = 3;
	/* Threshold for ignore contours on hand layer 1 */
	const double AREA_CONTOUR_THRESHOLD = 15;

	/* Minimum distance for merge finger layer 2 to contour layer 1 */
	const int MIN_DIST_12 = -4;

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
	vector<cv::Point> handBounder;

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
	double calDistance(cv::Point p1, cv::Point p2);
	cv::Point calCentroid(vector<cv::Point> points);
	double calAngle(cv::Point ph, cv::Point pi, cv::Point pj);

	void captureFrame();
private:
	KinectReader kinectReader;
	void calculateContourArea(vector<cv::Point> contour, double *area);
	int performKeyboardEvent(int key);

	struct ConvexSorter {
		cv::Point origin;
		bool operator() (cv::Point p1, cv::Point p2) {
			cv::Point pi = cv::Point(p1.x - origin.x, p1.y - origin.y);
			cv::Point pj = cv::Point(p2.x - origin.x, p2.y - origin.y);
			double pi_r = sqrt(pow(pi.x, 2) + pow(pi.y, 2));
			double pj_r = sqrt(pow(pj.x, 2) + pow(pj.y, 2));
			double pi_t, pj_t;
			if (pi.x == 0 && pi.y == 0) {
				pi_t = 0;
			}
			else if (pi.x >= 0) {
				pi_t = asin(pi.y / pi_r);
			}
			else {
				pi_t = -asin(pi.y / pi_r) + PI;
			}

			if (pj.x == 0 && pj.y == 0) {
				pj_t = 0;
			}
			else if (pj.x >= 0) {
				pj_t = asin(pj.y / pj_r);
			}
			else {
				pj_t = -asin(pj.y / pj_r) + PI;
			}

			return (pi_t < pj_t) || ((pi_t == pj_t) && (pi_r < pj_r));
		}
	};
};