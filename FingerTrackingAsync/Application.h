#pragma once
#include "KinectReader.h"
#include "AdapterCaller.h"
#define _USE_MATH_DEFINES
#include <cmath>
#define PI 3.14159265
#define INF HUGE_VAL

class Application {
public :
	int initialize();
	void start();

	const int FINGER_THUMB = 0;
	const int FINGER_INDEX = 1;
	const int FINGER_MIDDLE = 2;
	const int FINGER_RING = 3;
	const int FINGER_LITTLE = 4;
	const int PALM_POSITION = 5;
	const int PALM_ANGLE = 6;

	void setAdapterCaller(AdapterCaller adapterCaller);
protected :
	AdapterCaller adapterCaller;

	ushort maxDepth = 65535;
	ushort minDepth = 0;

	/* Percentage threshold for build hand layer 1 */
	const double BUILD_LAYER_1_THRESHOLD = 0.2;
	/* Percentage threshold for build hand layer 2 */
	const double BUILD_LAYER_2_THRESHOLD = 0.6;
	/* Percentage threshold for build hand layer 3 */
	const double BUILD_LAYER_3_THRESHOLD = 0.9;

	/* Threshold for select convexity defect on hand layer 2 */
	const int CONVEX_DEPTH_THRESHOLD_LAYER_2 = 10000;
	
	/* Threshold for select convexity defect on hand layer 3 */
	const int CONVEX_DEPTH_THRESHOLD_LAYER_3 = 4000;

	/* General threshold for cluster points */
	const int DISTANCE_THESHOLD = 10;
	/* Threshold for cluster corners on hand layer 1 */
	const int DISTANCE_THRESHOLD_CORNER_LAYER_1 = 3;
	/* Threshold for ignore contours on hand layer 1 */
	const double AREA_CONTOUR_THRESHOLD = 200;
	/* Threshold for select corner pixel on hand layer 1 */
	const int CORNER_THRESHOLD = 130;

	/* Minimum distance for merge finger layer 2 to contour layer 1 */
	const int MIN_DIST_12 = -4;

	const int HAND_RADIUS_MM = 50;

	const double ASSIGN_FINGER_ANGLE_THRESHOLD = (5 * PI / 180);

	const int SCAN_PALM_PADDING = 5;
	const int SCAN_PALM_EXCEPT_HOLE = 2;
	const int SCAN_TABLE_EXCEPT_HOLE = 4;

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

	map<int, cv::Point> finger2dMap;
	map<int, cv::Point3f> finger3dMap; // deprecate
	vector<cv::Point3f> finger3ds = vector<cv::Point3f>(7, cv::Point3f(0, 0, 0));

	cv::Mat skinMask;
	cv::Mat colorFrame;
	cv::Mat depthFrame;
	cv::Mat rawDepthFrame;
	cv::Mat edgeColorFrame;
	cv::Mat edgeMask = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	cv::Mat histogramFrame;
	vector<cv::Mat> prevHandMasks = vector<cv::Mat>(10, cv::Mat::zeros(cv::Size(640, 480), CV_8UC1));
	cv::Mat handMask;
	cv::Mat handLayer1Depth = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer1 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer2 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer3 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayer3_2 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat handLayerPalm = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	cv::Mat handLayerCut = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	cv::Mat handLayerAbs = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	cv::Mat palmMask = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	cv::Mat cutMask = cv::Mat(cv::Size(640, 480), CV_8UC1, cv::Scalar(255));
	cv::Point handPoint;
	int handRadius;


	cv::Point palmPoint = cv::Point(0, 0);
	cv::Point3f palmPoint3d;
	vector<vector<cv::Point>> contoursL1;
	map<int, vector<cv::Point>> cornerGroup;
	vector<cv::Point> fingerPoints;  

	
	cv::Rect palmRect;
	vector<cv::Point> extendedFinger;
	cv::Vec2d handDirection;

	void buildDepthHandMask();
	void buildHand3Layers();

	void evaluateHandLayer1();
	void evaluateHandLayer2();
	void evaluateHandLayer3();
	void evaluateHandLayerCut();
	void evaluate3Layer();
	void evaluatePalmAngle();

	void assignFingerId();

	void clusterPoint(vector<cv::Point>& inputArray, vector<cv::Point>& outputArray, int thresh);
	double calDistance(cv::Point p1, cv::Point p2);
	cv::Point calCentroid(vector<cv::Point> points);
	double calAngle(cv::Point ph, cv::Point pi, cv::Point pj);
	cv::Vec2d calLinear(cv::Point p1, cv::Point p2);
	cv::Point calInterceptPoint(cv::Vec2d l1, cv::Vec2d l2);
	cv::Vec2d calPerpendicularLine(cv::Vec2d l, cv::Point p);
	cv::Vec2d calParalellLine(cv::Vec2d l, cv::Point p);
	void calEndpoint(cv::Vec2d l, cv::Point &p1, cv::Point &p2);
	cv::Point calMedianPoint(cv::Point p1, cv::Point p2);
	cv::Point calRatioPoint(cv::Point p1, cv::Point p2, double ratio1, double ratio2);
	cv::Point calLinearPointByX(cv::Vec2d L, double x);
	cv::Point calLinearPointByY(cv::Vec2d L, double y);
	void calLinearInterceptCirclePoint(cv::Point center, double radius, cv::Vec2d linear, cv::Point &p_out1, cv::Point &p_out2);
	cv::Point2d convertPointCartesianToPolar(cv::Point p, cv::Point o = cv::Point(0, 0));
	cv::Point3f convertPoint2dTo3D(cv::Point p);
	cv::Point calRadiusPoint(double angle, double radius, cv::Point origin);
	static double calAnglePoint(cv::Point origin, cv::Point p);
	double calLinerAngleByPoint(cv::Vec2d l, cv::Point p);

	vector<cv::Point> findLargestContour(cv::Mat in);
	vector<cv::Point> findConcavePoints(vector<cv::Point> contour, vector<int> hull, int threshold);

	void sendData();

	void captureFrame();
private:
	KinectReader kinectReader;

	cv::Point EL3_findMinConcave(vector<cv::Point> concavePoints);
	void EL3_buildTable(cv::Mat in, cv::Rect boundingBox, vector<bool>& acceptTransitionTable, vector<bool>& acceptLengthTable);
	cv::Vec2i EL3_findMaxRegion(vector<bool> acceptTransitionTable, vector<bool> acceptLengthTable);
	cv::Point EL3_findRegionCenter(cv::Mat in, cv::Rect region);
	int EL3_countWhitePoint(cv::Mat in, cv::Point point, int radius);
	
	
	void calculateContourArea(vector<cv::Point> contour, double *area);
	int performKeyboardEvent(int key);

	struct FingerSorter {
		cv::Point origin = cv::Point(0, 0);
		bool operator() (cv::Point p1, cv::Point p2) {
			double t1 = calAnglePoint(origin, p1);
			double t2 = calAnglePoint(origin, p2);
			return t1 < t2;
		}
	};

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