#include "pch.h"
#include "Application.h"
#include "KinectReader.h"
#include "opencv2/opencv.hpp"
#include "OpenCVThreadFactory.h"
#include <algorithm>

using namespace std;

int Application::initialize()
{
	int status = 0;
	status = kinectReader.initialize();
	if (status != 0)
		return 1;

	cv::namedWindow(WINDOW_RGB, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_DEPTH, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_MASK_L1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_MASK_L2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_MASK_L3, cv::WINDOW_AUTOSIZE);
	return 0;
}

void Application::start()
{
	while (true) {
		kinectReader.readRGBFrame();
		kinectReader.readDepthFrame();
		colorFrame = kinectReader.getRGBFrame();
		depthFrame = kinectReader.getDepthFrame();
		rawDepthFrame = kinectReader.getRawDepthFrame();
		
		if (kinectReader.isHandTracking()) {
			thread buildDepthHandMaskT = thread(&Application::buildDepthHandMask, this);

			thread transformColorFrameT = thread(&Application::transformColorFrame, this);
			transformColorFrameT.join(); // 13ms
			//thread buildSkinMaskT = thread(&Application::buildSkinMask, this);
			thread buildEdgeColorT = thread(&Application::buildEdgeColor, this);
			
			//buildSkinMaskT.join(); // 18ms
			buildDepthHandMaskT.join();	// ~8 from start 
			//buildHistogram();

			//thread combineSkinDepthT = thread(&Application::combineSkinHandMask, this);
			//combineSkinDepthT.join(); // 7ms

			thread buildHand3LayersT = thread(&Application::buildHand3Layers, this);
			buildHand3LayersT.join(); //10

			//buildHistogram();
			buildRawHandHistogram();
			int a = 0;
			thread evaluateHandLayer1T = thread(&Application::evaluateHandLayer1, this);
			thread evaluateHandLater2T = thread(&Application::evaluateHandLayer2, this);

			evaluateHandLater2T.join();
			evaluateHandLayer1T.join();

			evaluateLayer12();

			/*evaluateHandLayer1();
			evaluateHandLater2();*/
			int b = 0;
			buildEdgeColorT.join();		

			cv::imshow(WINDOW_MASK_L1, handLayer1); // 14
			cv::imshow(WINDOW_MASK_L2, handLayer2);
			cv::imshow(WINDOW_MASK_L3, handLayer3);
			cv::imshow("Edge", edgeColorFrame);
			cv::imshow("histogram", histogramFrame);
		}

		if (tickCount == 0) {
			tickCount = cv::getTickCount();
		}
		else {
			int64 t = cv::getTickCount();
			double fpsT = cv::getTickFrequency() / (t - tickCount);
			tickCount = t;
			cout << "FPS " << fpsT << endl;
		}

		//cv::normalize(rawDepthFrame, rawDepthFrame, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imshow(WINDOW_RGB, colorFrame);
		cv::imshow(WINDOW_DEPTH, depthFrame);

		int key = cv::waitKey(1);
		if (performKeyboardEvent(key) != 0) {
			break;
		}
	}
	
	cv::destroyAllWindows();
}

void Application::transformColorFrame()
{
	cv::Mat roi = colorFrame.clone()(cv::Rect(cv::Point(48, 56), cv::Size(577, 424)));
	cv::resize(roi, roi, cv::Size(640, 470));
	cv::Mat blackRow = cv::Mat::zeros(cv::Size(640, 10), CV_8UC3);
	roi.push_back(blackRow);
	roi.copyTo(colorFrame);
}

void Application::buildEdgeColor()	// ~45ms
{
	float handPosX = kinectReader.getHandPosX();
	float handPosY = kinectReader.getHandPosY();
	cv::Mat gray;
	cv::cvtColor(colorFrame, gray, cv::COLOR_BGR2GRAY);
	//cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	cv::Canny(gray, edgeColorFrame, 0, 150, 3);
	//cv::dilate(edgeColorFrame, edgeColorFrame, cv::Mat());
	//cv::bitwise_not(edgeColorFrame, edgeColorFrame);
	//cv::floodFill(edgeColorFrame, cv::Point(handPosX, handPosY), cv::Scalar(128));
	//cv::circle(edgeColorFrame, cv::Point(handPosX, handPosY), 4, cv::Scalar(0, 0, 0), -1);
}

void Application::buildSkinMask()
{
	cv::Mat colorFrameHSV;
	cv::cvtColor(colorFrame, colorFrameHSV, cv::COLOR_BGR2HSV);
	cv::Mat semi_skinMask_1, semi_skinMask_2;
	int b = 0;
	thread semi_skinMask_1_thread = OpenCVThreadFactory::inRange(colorFrameHSV, cv::Scalar(160, 10, 60), cv::Scalar(179, 255, 255), semi_skinMask_1);
	thread semi_skinMask_2_thread = OpenCVThreadFactory::inRange(colorFrameHSV, cv::Scalar(0, 10, 60), cv::Scalar(40, 150, 255), semi_skinMask_2);

	semi_skinMask_1_thread.join();
	semi_skinMask_2_thread.join();

	thread skinMask_thread = OpenCVThreadFactory::bitwise_or(semi_skinMask_1, semi_skinMask_2, skinMask);

	skinMask_thread.join();
	int a = 0;
	//cv::inRange(colorFrameHSV, cv::Scalar(160, 10, 60), cv::Scalar(179, 255, 255), semi_skinMask_1);
	//cv::inRange(colorFrameHSV, cv::Scalar(0, 10, 60), cv::Scalar(40, 150, 255), semi_skinMask_2);
	//cv::bitwise_or(semi_skinMask_1, semi_skinMask_2, skinMask);
	int c = 0;
}

void Application::buildDepthHandMask()
{
	kinectReader.getDepthHandMask().copyTo(handMask);
	float handPosX = kinectReader.getHandPosX();
	float handPosY = kinectReader.getHandPosY();
	int smallKernel = 3;
	for (int y = handPosY - smallKernel; y < handPosY + smallKernel; y++) {
		for (int x = handPosX - smallKernel; x < handPosX + smallKernel; x++) {
			handMask.at<uchar>(y, x) = 128;
		}
	}
	cv::floodFill(handMask, cv::Point((int)handPosX, (int)handPosY), cv::Scalar(255));
	cv::threshold(handMask, handMask, 129, 255, cv::THRESH_BINARY);
}

void Application::combineSkinHandMask()
{
	cv::Mat mask;
	cv::bitwise_and(skinMask, handMask, mask);
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	cv::Mat black = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	black.copyTo(handMask);
	
	if (contours.size() == 0) {
		return;
	}

	vector<double> areas(contours.size());

	for (int i = 0; i < contours.size(); i++) {
		areas[i] = cv::contourArea(contours[i]);
	}

	double maxArea = 0;
	int largestIndex = 0;
	for (int i = 0; i < areas.size(); i++) {
		if (areas[i] > maxArea) {
			maxArea = areas[i];
			largestIndex = i;
		}
	}
	cv::drawContours(handMask, contours, largestIndex, cv::Scalar(255, 255, 255), -1, 8, vector<cv::Vec4i>(), 0, cv::Point(0, 0));
}

void Application::buildHand3Layers()
{

	minDepth = 65535;
	maxDepth = 0;
	int count = 0;
	for (int i = 0; i < rawDepthFrame.rows; i++) {
		ushort* rawRow = rawDepthFrame.ptr<ushort>(i);
		uchar* maskRow = handMask.ptr<uchar>(i);
		for (int j = 0; j < rawDepthFrame.cols; j++) {
			if (maskRow[j] != 0) {
				if (rawRow[j] > maxDepth)
					maxDepth = rawRow[j];
				if (rawRow[j] < minDepth && rawRow[j] != 0)
					minDepth = rawRow[j];
				count++;
			}
		}
	}
	int range = maxDepth - minDepth;
	int l1_max = minDepth + (range * BUILD_LAYER_1_THRESHOLD);
	int l2_max = minDepth + (range * BUILD_LAYER_2_THRESHOLD);
	int l3_max = minDepth + (range * BUILD_LAYER_3_THRESHOLD);

	cv::threshold(rawDepthFrame, handLayer1, l1_max, 65535, cv::THRESH_BINARY_INV);
	handLayer1.convertTo(handLayer1, CV_8UC1, 255.0 / 65535);
	cv::bitwise_and(handMask, handLayer1, handLayer1);

	cv::threshold(rawDepthFrame, handLayer2, l2_max, 65535, cv::THRESH_BINARY_INV);
	handLayer2.convertTo(handLayer2, CV_8UC1, 255.0 / 65535);
	cv::bitwise_and(handMask, handLayer2, handLayer2);

	cv::threshold(rawDepthFrame, handLayer3, l3_max, 65535, cv::THRESH_BINARY_INV);
	handLayer3.convertTo(handLayer3, CV_8UC1, 255.0 / 65535);
	cv::bitwise_and(handMask, handLayer3, handLayer3);
}

void Application::buildHistogram()
{
	vector<int> histogram(65536, 0);
	int minDepth = 65536;
	int maxDepth = 0;
	for (int i = 0; i < rawDepthFrame.rows; i++) {
		ushort* rawRow = rawDepthFrame.ptr<ushort>(i);
		uchar* maskRow = handLayer3.ptr<uchar>(i);
		for (int j = 0; j < rawDepthFrame.cols; j++) {
			if (maskRow[j] != 0) {
				if (rawRow[j] > maxDepth)
					maxDepth = rawRow[j];
				if (rawRow[j] < minDepth && rawRow[j] != 0)
					minDepth = rawRow[j];
				histogram[rawRow[j]] += 1;
			}
		}
	}

	int maxHist = 0;
	for (int i = minDepth; i <= maxDepth; i++) {
		if (histogram[i] > maxHist) {
			maxHist = histogram[i];
		}
	}
	cv::Mat histogramImage = cv::Mat::zeros(cv::Size(400, 200), CV_8UC1);
	for (int i = minDepth; i < maxDepth; i++) {
		int h = (((double)histogram[i]) / maxHist) * 200;
		int x = (i - minDepth) * 2;
		cv::rectangle(histogramImage, cv::Rect(cv::Point(x, 200 - h), cv::Size(2, h)), cv::Scalar(255), -1);
	}
	histogramImage.copyTo(histogramFrame);

}

void Application::buildRawHandHistogram()
{
	vector<int> histogram(65536, 0);
	int handDepth = kinectReader.getHandDepth();
	int range = kinectReader.getDepthHandRange();
	int minDepth = 65536;
	int maxDepth = 0;
	for (int i = 0; i < rawDepthFrame.rows; i++) {
		ushort* rawRow = rawDepthFrame.ptr<ushort>(i);
		uchar* maskRow = handMask.ptr<uchar>(i);
		for (int j = 0; j < rawDepthFrame.cols; j++) {
			if (maskRow[j] != 0) {
				if (rawRow[j] > maxDepth)
					maxDepth = rawRow[j];
				if (rawRow[j] < minDepth && rawRow[j] != 0)
					minDepth = rawRow[j];
				histogram[rawRow[j]] += 1;
			}
		}
	}

	const double MAX_SCALE_DEPTH = 3000;
	cv::Mat histogramImage = cv::Mat::zeros(cv::Size(402, 200), CV_8UC3);
	for (int i = handDepth - range; i <= handDepth + range && i >= 0 && i <= 65535; i++) {
		int h = (histogram[i] / MAX_SCALE_DEPTH) * 200;
		int x = (i - (handDepth - range)) * 2;
		cv::rectangle(histogramImage, cv::Rect(cv::Point(x, 200 - h), cv::Size(2, h)), cv::Scalar(255, 255, 255), -1);
		if (i == handDepth) {
			cv::line(histogramImage, cv::Point(x, 0), cv::Point(x, 200), cv::Scalar(0, 0, 255), 1);
		}
	}
	histogramImage.copyTo(histogramFrame);
}

void Application::evaluateHandLayer1() // ~66ms
{
	handLayer1Corners.clear();
	cv::Mat corner;
	cv::cornerHarris(handLayer1, corner, 8, 5, 0.04, cv::BORDER_DEFAULT);
	cv::normalize(corner, corner, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	for (int j = 0; j < corner.rows; j++) {
		float* cornerRow = corner.ptr<float>(j);
		for (int i = 0; i < corner.cols; i++) {
			if (cornerRow[i] > 160) {
				if (handLayer1.ptr<uchar>(j)[i] > 0)
					handLayer1Corners.push_back(cv::Point(i, j));
			}
		}
	}

	// group corner by contours
	vector<cv::Vec4i> hierachy;
	cv::findContours(handLayer1, contoursL1, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cornerGroup.clear();
	for (int i = 0; i < handLayer1Corners.size(); i++) {
		cv::Point corner_i = handLayer1Corners[i];
		for (int j = 0; j < contoursL1.size(); j++) {
			if (cv::pointPolygonTest(contoursL1[j], corner_i, false) > 0) {
				if (cornerGroup.count(j) == 0) {
					cornerGroup[j] = vector<cv::Point>();
				}
				cornerGroup[j].push_back(corner_i);
			}
		}
	}

	// cluster very near corner
	cv::cvtColor(handLayer1, handLayer1, cv::COLOR_GRAY2BGR);
	vector<int> ignoreContours;
	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		double area = cv::contourArea(contoursL1[it->first]);
		if (area < AREA_CONTOUR_THRESHOLD) {
			ignoreContours.push_back(it->first);
			continue;
		}
	}
	for (int i = 0; i < ignoreContours.size(); i++) {
		cornerGroup.erase(ignoreContours[i]);
	}
	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		vector<cv::Point> cluster;
		clusterPoint(it->second, cluster, DISTANCE_THRESHOLD_CORNER_LAYER_1);
		cornerGroup[it->first] = cluster;
	}
	
	
	for (int i = 0; i < handLayer1Corners.size(); i++) {
		cv::circle(handLayer1, handLayer1Corners[i], 1, cv::Scalar(0, 0, 255), -1);
	}
}

void Application::evaluateHandLayer2()	// 7ms
{
	cv::Mat handLayer2Copy;
	handLayer2.copyTo(handLayer2Copy);

	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierachy;
	cv::findContours(handLayer2, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
	vector<vector<cv::Point>> convexHull(contours.size());
	vector<vector<int>> convexHullI(contours.size());
	double largestArea = 0;
	int largestIndex = 0;
	for (int i = 0; i < contours.size(); i++) {
		cv::convexHull(contours[i], convexHull[i]);
		cv::convexHull(contours[i], convexHullI[i]);

		double a = cv::contourArea(contours[i]);
		if (a > largestArea) {
			largestArea = a;
			largestIndex = i;
		}
	}
	if (largestArea == 0)
		return;

	hullL2.clear();
	for (int i = 0; i < convexHull[largestIndex].size(); i++) {
		hullL2.push_back(convexHull[largestIndex][i]);
	}

	vector<cv::Point> semi_fingerPoint, abyss_finger;
	cv::cvtColor(handLayer2, handLayer2, cv::COLOR_GRAY2BGR);
	vector<cv::Vec4i> defect;
	cv::convexityDefects(contours[largestIndex], convexHullI[largestIndex], defect);
	bool firstF = true;
	for (int i = 0; i < defect.size(); i++) {
		cv::Vec4i v = defect[i];
		int depth = v[3];
		if (depth > CONVEX_DEPTH_THRESHOLD_LAYER_2) {
			cv::Point startPoint = contours[largestIndex][v[0]];
			cv::Point endPoint = contours[largestIndex][v[1]];
			cv::Point farPoint = contours[largestIndex][v[2]];

			double ms = ((double)(startPoint.y - farPoint.y)) / (startPoint.x - farPoint.x);
			double me = ((double)(endPoint.y - farPoint.y)) / (endPoint.x - farPoint.x);
			double angle = atan((me - ms) / (1 + (ms * me))) * (180 / PI);

			if (angle < 0) {
				semi_fingerPoint.push_back(startPoint);
				semi_fingerPoint.push_back(endPoint);
				abyss_finger.push_back(farPoint);
				cv::circle(handLayer2, farPoint, 4, cv::Scalar(255, 255, 0), 2);
				/*cv::circle(handLayer2, startPoint, 2, cv::Scalar(0, 255, 0), 1);
				cv::circle(handLayer2, endPoint, 2, cv::Scalar(0, 255, 0), 1);*/
				cout << "depth abyss " << depth << endl;
			}
		}
	}
	
	clusterPoint(semi_fingerPoint, fingerL2Point, DISTANCE_THESHOLD);
	double sum_x = 0;
	double sum_y = 0;
	int count = 0;
	for (int i = 0; i < fingerL2Point.size(); i++) {
		cv::circle(handLayer2, fingerL2Point[i], 4, cv::Scalar(0, 255, 0), 2);
		sum_x += fingerL2Point[i].x;
		sum_y += fingerL2Point[i].y;
		count += 1;
	}
	if (count == 0)
		return;

	// find finger centroid point
	cv::Point fingerCentroid;
	fingerCentroid.x = sum_x / count;
	fingerCentroid.y = sum_y / count;
	cv::circle(handLayer2, fingerCentroid, 4, cv::Scalar(100, 255, 100), -1);

	double maxAbyssDist1 = 0;
	double maxAbyssDist2 = 0;
	int maxAbyssIndex1 = -1;
	int maxAbyssIndex2 = -1;
	for (int i = 0; i < abyss_finger.size(); i++) {
		double d = calDistance(fingerCentroid, abyss_finger[i]);
		if (d > maxAbyssDist1) {
			maxAbyssDist2 = maxAbyssDist1;
			maxAbyssIndex2 = maxAbyssIndex1;
			maxAbyssDist1 = d;
			maxAbyssIndex1 = i;
		}
		else if (d > maxAbyssDist2) {
			maxAbyssDist2 = d;
			maxAbyssIndex2 = i;
		}
	}
	if (maxAbyssIndex1 != -1)
		cv::circle(handLayer2, abyss_finger[maxAbyssIndex1], 4, cv::Scalar(255, 255, 0), -1);
	if (maxAbyssIndex2 != -1)
		cv::circle(handLayer2, abyss_finger[maxAbyssIndex2], 4, cv::Scalar(255, 0, 255), -1);

	handBounder.clear();
	
	for (int i = 0; i < abyss_finger.size(); i++) {
		int k = (maxAbyssIndex1 + i) % abyss_finger.size();
		int j = (maxAbyssIndex1 + i + 1) % abyss_finger.size();

		handBounder.push_back(abyss_finger[i]);

		if (abyss_finger.size() > 2 && ((k == maxAbyssIndex1 && j == maxAbyssIndex2) || (k == maxAbyssIndex2 && j == maxAbyssIndex1)))
			continue;
		cv::Point pk = abyss_finger[k];
		cv::Point pj = abyss_finger[j];
		cv::Point p, q, pp, qq;

		int p_index = 0;
		int q_index = 0;
		
		if (k == maxAbyssIndex1 || k == maxAbyssIndex2) {
			p = pk;
			q = pj;
			p_index = k;
			q_index = j;
		}
		else if (j == maxAbyssIndex1 || j == maxAbyssIndex2) {
			p = pj;
			q = pk;
			p_index = j;
			p_index = k;
		}
		else {
			cv::line(handLayer2Copy, pk, pj, cv::Scalar(0, 0, 0), 2);
			continue;
		}			

		double dx = p.x - q.x;
		double dy = p.y - q.y;

		int MOVE_DISTANCE = 100;
		if (abyss_finger.size() == 2) {
			if (dx > 0) {
				if (dy != 0) {
					pp.x = 640;
					pp.y = 640 * (dy / dx) - (dy / dx) * p.x + p.y;
					qq.x = 0;
					qq.y = qq.y - (dy / dx) * qq.x;
				}
				else {
					pp.x = 640;
					pp.y = p.y;
					qq.x = 0;
					qq.y = q.y;
				}
			}
			else if (dx < 0) {
				if (dy != 0) {
					pp.x = 0;
					pp.y = p.y - (dy / dx) * p.x;
					qq.x = 640;
					qq.y = 640 * (dy / dx) - (dy / dx)*q.x + q.y;
				}
				else {
					pp.x = 0;
					pp.y = p.y;
					qq.x = 640;
					qq.y = q.y;
				}
			}
			else {
				if (dy > 0) {
					pp.x = p.x;
					pp.y = 0;
					qq.x = q.x;
					qq.y = 480;
				}
				else {
					pp.x = p.x;
					pp.y = 480;
					qq.x = q.x;
					qq.y = 0;
				}
			}

			handBounder.push_back(pp);
			handBounder.push_back(qq);

			cv::line(handLayer2Copy, pp, qq, cv::Scalar(0, 0, 0), 2);
			cv::line(handLayer2, pp, qq, cv::Scalar(255, 0, 0), 5);
			break;
		}

		if (dx > 0) {
			pp.x = 640;
			if (dy != 0) {
				pp.y = 640 * (dy / dx) - (dy / dx) * p.x + p.y;
			}
			else {
				pp.y = p.y;
			}
		}
		else if (dx < 0) {
			pp.x = 0;
			if (dy != 0) {
				pp.y = p.y - p.x * (dy / dx);
			}
			else {
				pp.y = p.y;
			}
		}
		else {
			pp.x = p.x;
			if (dy > 0) {
				pp.y = 480;
			}
			else {
				pp.y = 0;
			}
		}

		handBounder.push_back(pp);

		cv::line(handLayer2Copy, pp, q, cv::Scalar(0, 0, 0), 2);
		cv::line(handLayer2, pp, q, cv::Scalar(255, 0, 0), 5);
	}
	handBounder.push_back(cv::Point(0, 480));
	handBounder.push_back(cv::Point(640, 480));

	cv::Point centroidHandConvex = calCentroid(handBounder);
	ConvexSorter handConvexSorter;
	handConvexSorter.origin = centroidHandConvex;
	sort(handBounder.begin(), handBounder.end(), handConvexSorter);

	cv::cvtColor(handLayer3, handLayer3, cv::COLOR_GRAY2BGR);
	cv::drawContours(handLayer3, vector<vector<cv::Point>> { handBounder }, 0, cv::Scalar(0, 255, 255), 3);

	vector<vector<cv::Point>> contoursCopy;
	vector<cv::Vec4i> hierachyCopy;
	cv::findContours(handLayer2Copy, contoursCopy, hierachyCopy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	double largestAreaCopy = 0;
	double largestIndexCopy = -1;
	for (int i = 0; i < contoursCopy.size(); i++) {
		double a = cv::contourArea(contoursCopy[i]);
		if (a > largestAreaCopy) {
			largestAreaCopy = a;
			largestIndexCopy = i;
		}
	}

	if (largestIndexCopy != -1) {
		cv::drawContours(handLayer2, contoursCopy, largestIndexCopy, cv::Scalar(128, 128, 128), -1);
	}
}

void Application::evaluateHandLayer3()
{
	// TODO : perform on layer 3
	
}

void Application::evaluateLayer12()
{
	vector<cv::Point> fingerL1Point;
	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		vector<cv::Point> group = it->second;
		vector<cv::Point> contourL1 = contoursL1[it->first];
		bool ignore = false;
		for (int i = 0; i < fingerL2Point.size(); i++) {
			double d = cv::pointPolygonTest(contourL1, fingerL2Point[i], true);
			if (d >= MIN_DIST_12) {
				/*cv::circle(handLayer2, fingerL2Point[i], 6, cv::Scalar(0, 0, 255), 2);
				cv::drawContours(handLayer2, contoursL1, it->first, cv::Scalar(0, 0, 255), 2);*/
				group.push_back(fingerL2Point[i]);
				//ignore = true;
				break;
			}
		}
		/*if (ignore)
			continue;*/

		/*for (int i = 0; i < group.size(); i++) {
			cv::Point corner = group[i];
			for (int j = 0; j < fingerL2Point.size(); j++) {
				cv::Point finger = fingerL2Point[j];
				double dist = calDistance(finger, corner);
				if (dist < DISTANCE_THESHOLD) {
					ignore = true;
				}
			}
		}

		if (ignore)
			continue;
		double farthestDist = 0;
		int farthestIndex = 0;
		for (int i = 0; i < group.size(); i++) {
			double d = cv::pointPolygonTest(hullL2, group[i], true);
			if (d > farthestDist) {
				farthestDist = d;
				farthestIndex = i;
			}
		}

		fingerL1Point.push_back(group[farthestIndex]);*/
	}

	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		vector<cv::Point> points = it->second;
		cv::Point centroid = calCentroid(points);
		ConvexSorter sorter;
		sorter.origin = centroid;
		sort(points.begin(), points.end(), sorter);
		for (int i = 0; i < points.size(); i++) {
			cv::Point pi = points[i];
			cv::Point pj = points[(i + 1) % points.size()];
			cv::circle(handLayer2, pi, 2, cv::Scalar(0, 0, 255), -1);
			cv::line(handLayer2, pi, pj, cv::Scalar(0, 0, 255), 1);
		}
	}
	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		if (handBounder.size() == 0)
			break;

		vector<cv::Point> points = it->second;
		bool isInside = false;
		double minDist = 1000000;
		double maxDist = -1000000;
		int minIndex = -1;
		int maxIndex = -1;
		for (int i = 0; i < points.size(); i++) {
			double d = cv::pointPolygonTest(handBounder, points[i], true);

			if (d >= 0)
				isInside = true;
			if (d > maxDist) {
				maxDist = d;
				maxIndex = i;
			}
			if (d < minDist) {
				minDist = d;
				minIndex = i;
			}
		}

		if (isInside) {
			cv::circle(handLayer2, points[maxIndex], 4, cv::Scalar(0, 255, 255), 2);
		}
		else {
			cv::circle(handLayer2, points[minIndex], 4, cv::Scalar(0, 255, 255), 2);
		}
	}
	/*for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		vector<cv::Point> points = it->second;
		if (points.size() == 1) {
			cv::circle(handLayer2, points[0], 4, cv::Scalar(0, 255, 255), 2);
		}
		else if (points.size() == 2) {

		}
		else {
			double minDegree = 360;
			int minIndex = -1;
			for (int i = 0; i < points.size(); i++) {
				cv::Point ph = points[(i - 1) % points.size()];
				cv::Point pi = points[i];
				cv::Point pj = points[(i + 1) % points.size()];

				double angle = abs(calAngle(ph, pi, pj));
				if (angle < minDegree) {
					minDegree = angle;
					minIndex = i;
				}
			}

			cv::circle(handLayer2, points[minIndex], 4, cv::Scalar(0, 255, 255), 2);
		}
	}*/
	/*for (int i = 0; i < fingerL1Point.size(); i++) {
		cv::circle(handLayer2, fingerL1Point[i], 4, cv::Scalar(0, 0, 255), 2);
	} */
}

void Application::clusterPoint(vector<cv::Point>& inputArray, vector<cv::Point>& outputArray, int thresh)
{
	outputArray.clear();
	vector<vector<double>> distMat(inputArray.size());
	for (int i = 0; i < distMat.size(); i++) {
		cv::Point pi = inputArray[i];
		distMat[i] = vector<double>(inputArray.size());
		for (int j = i + 1; j < distMat[i].size(); j++) {
			cv::Point pj = inputArray[j];
			distMat[i][j] = calDistance(pi, pj);
		}
	}

	vector<int> cluster(distMat.size(), 0);
	int clusterCount = 1;
	for (int i = 0; i < distMat.size(); i++) {
		for (int j = i + 1; j < distMat[i].size(); j++) {
			if (distMat[i][j] < thresh) {
				if (cluster[i] == 0 && cluster[j] == 0) {
					cluster[i] = clusterCount;
					cluster[j] = clusterCount;
					clusterCount++;
				}
				else if (cluster[i] == 0) {
					cluster[i] = cluster[j];
				}
				else {
					cluster[j] = cluster[i];
				}
			}
		}
	}

	for (int i = 0; i < cluster.size(); i++) {
		if (cluster[i] == 0) {
			cluster[i] = clusterCount;
			clusterCount++;
		}
	}

	for (int i = 1; i < clusterCount; i++) {
		// find centroid of cluster i
		double sum_x = 0;
		double sum_y = 0;
		int count = 0;
		for (int j = 0; j < cluster.size(); j++) {
			if (cluster[j] == i) {
				sum_x += inputArray[j].x;
				sum_y += inputArray[j].y;
				count++;
			}
		}
		if (count == 0)
			continue;
		cv::Point pi = cv::Point(sum_x / count, sum_y / count);
		outputArray.push_back(pi);
	}

}

double Application::calDistance(cv::Point p1, cv::Point p2)
{
	double d = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
	return d;
}

cv::Point Application::calCentroid(vector<cv::Point> points)
{
	double sum_x = 0;
	double sum_y = 0;
	int count = 0;
	for (int i = 0; i < points.size(); i++) {
		sum_x += points[i].x;
		sum_y += points[i].y;
		count += 1;
	}
	if (count == 0) 
		return cv::Point();
	return cv::Point(sum_x / count, sum_y / count);
}

double Application::calAngle(cv::Point ph, cv::Point pi, cv::Point pj)
{
	double ms = ((double)(ph.y - pi.y)) / (ph.x - pi.x);
	double me = ((double)(pj.y - pi.y)) / (pj.x - pi.x);
	double angle = atan((me - ms) / (1 + (ms * me))) * (180 / PI);
	return angle;
}

void Application::captureFrame()
{
	time_t ts = time(nullptr);
	
	//char buffer_hist[80];
	//sprintf_s(buffer_hist, "%d - histogram.jpg", ts);
	//cv::imwrite(buffer_hist, histogramFrame);
	//
	//char buffer_mask3[80];
	//sprintf_s(buffer_mask3, "%d -  hand layer 3.jpg", ts);
	//cv::imwrite(buffer_mask3, handLayer3);

	cv::Mat scaledHistogram, concat, rawHand, handMask16U, edgeFrameC3;
	//handMask.convertTo(handMask16U, CV_16UC1);
	cv::normalize(handMask, handMask16U, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
	cv::bitwise_and(rawDepthFrame, handMask16U, rawHand);
	cv::normalize(rawHand, rawHand, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::cvtColor(rawHand, rawHand, cv::COLOR_GRAY2BGR);
	cv::cvtColor(edgeColorFrame, edgeFrameC3, cv::COLOR_GRAY2BGR);
	cv::Mat row = cv::Mat::zeros(cv::Size(402, 280), CV_8UC3);
	histogramFrame.copyTo(scaledHistogram);
	scaledHistogram.push_back(row);

	cv::hconcat(scaledHistogram, rawHand, concat);
	cv::hconcat(concat, edgeFrameC3, concat);
	cv::rectangle(concat, cv::Rect(cv::Point(0, 0), cv::Size(402, 200)), cv::Scalar(255), 1);
	char buffer_concat[80];
	sprintf_s(buffer_concat, "%d - concated.jpg", ts);
	cv::imwrite(buffer_concat, concat);
}

void Application::calculateContourArea(vector<cv::Point> contour, double * area)
{
	(*area) = cv::contourArea(contour);
}

int Application::performKeyboardEvent(int key)
{
	if (key == 27) // esc
		return 1;
	else if (key == 'c' || key == 'C')
		captureFrame();
	return 0;
}
