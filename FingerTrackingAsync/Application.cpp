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

			handPoint = kinectReader.getHandPoint();
			handRadius = kinectReader.getHandRadius(HAND_RADIUS_MM);
			
			thread buildDepthHandMaskT = thread(&Application::buildDepthHandMask, this);

			//thread transformColorFrameT = thread(&Application::transformColorFrame, this);
			//transformColorFrameT.join(); // 13ms
			//thread buildSkinMaskT = thread(&Application::buildSkinMask, this);
			thread buildEdgeColorT = thread(&Application::buildEdgeColor, this);
			
			//buildSkinMaskT.join(); // 18ms
			buildDepthHandMaskT.join();	// ~8 from start 
			//transformDepthFrame();
			//buildHistogram();

			//thread combineSkinDepthT = thread(&Application::combineSkinHandMask, this);
			//combineSkinDepthT.join(); // 7ms

			thread buildHand3LayersT = thread(&Application::buildHand3Layers, this);
			buildHand3LayersT.join(); //10

			buildEdgeColorT.join();
			buildEdgeMask();

			cv::imshow("edge maskx", edgeMask);
			cv::imshow("hand layer1 depth", handLayer1Depth);

			//buildHistogram();
			//buildRawHandHistogram();

			evaluateHandLayerPalm();

			thread evaluateHandLayer1T = thread(&Application::evaluateHandLayer1, this);
			thread evaluateHandLater2T = thread(&Application::evaluateHandLayer2, this);

			evaluateHandLater2T.join();
			evaluateHandLayer1T.join();

			evaluateHandLayer3();
			evaluate3Layer();

			assignFingerId();

			//adapterCaller.sendData(finger3ds);
			thread(&Application::sendData, this).detach();
			/*evaluateHandLayer1();
			evaluateHandLater2();*/
				
			cv::imshow(WINDOW_MASK_L1, handLayer1); // 14
			cv::imshow(WINDOW_MASK_L2, handLayer2);
			cv::imshow(WINDOW_MASK_L3, handLayer3);
			cv::imshow("Hand Absolute", handLayerAbs);
			cv::imshow("Edge", edgeColorFrame);
			cv::imshow("Palm", handLayerPalm);
			//cv::imshow("histogram", histogramFrame);
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

void Application::setAdapterCaller(AdapterCaller adapterCaller)
{
	this->adapterCaller = adapterCaller;
}

void Application::transformColorFrame()
{
	cv::Mat roi = colorFrame.clone()(cv::Rect(cv::Point(56, 56), cv::Size(569, 424)));
	cv::resize(roi, roi, cv::Size(640, 470));
	cv::Mat blackRow = cv::Mat::zeros(cv::Size(640, 10), CV_8UC3);
	roi.push_back(blackRow);
	roi.copyTo(colorFrame);
}

void Application::transformDepthFrame()
{
	cv::Point2f src_verts[4];
	src_verts[0] = cv::Point(639, 479);
	src_verts[1] = cv::Point(0, 479);
	src_verts[2] = cv::Point(0, 0);
	src_verts[3] = cv::Point(639, 0);
	
	cv::Point2f dst_verts[4];
	dst_verts[0] = cv::Point(639, 479);
	dst_verts[1] = cv::Point(56, 479);
	dst_verts[2] = cv::Point(57, 48);
	dst_verts[3] = cv::Point(639, 51);

	cv::Mat m = cv::getPerspectiveTransform(src_verts, dst_verts);
	cv::warpPerspective(depthFrame, depthFrame, m, depthFrame.size());
	cv::warpPerspective(handMask, handMask, m, handMask.size());
}

void Application::buildEdgeColor()	// ~45ms
{
	float handPosX = kinectReader.getHandPosX();
	float handPosY = kinectReader.getHandPosY();
	cv::Mat gray;
	cv::cvtColor(colorFrame, gray, cv::COLOR_BGR2GRAY);
	//cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	cv::Canny(gray, edgeColorFrame, 50, 150, 3);
	cv::dilate(edgeColorFrame, edgeColorFrame, cv::Mat());
	cv::erode(edgeColorFrame, edgeColorFrame, cv::Mat());
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

	handMask.copyTo(handLayerAbs);
	cv::cvtColor(handLayerAbs, handLayerAbs, cv::COLOR_GRAY2BGR);

	cv::Mat maskLayer1;
	handLayer1.convertTo(maskLayer1, CV_16UC1);
	handLayer1.copyTo(handLayer1Depth);
	
	/*cv::bitwise_and(maskLayer1, rawDepthFrame, handLayer1Depth);
	cv::normalize(handLayer1Depth, handLayer1Depth, 0, 255, cv::NORM_MINMAX, CV_8UC1, handLayer1);*/
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierachy;
	
	cv::findContours(handLayer1Depth, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	vector<vector<int>> hull(contours.size());
	vector<vector<cv::Vec4i>> convex(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		cv::convexHull(contours[i], hull[i]);
		cv::convexityDefects(contours[i], hull[i], convex[i]);
	}
	cv::cvtColor(handLayer1Depth, handLayer1Depth, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < convex.size(); i++)
	{
		for (int j = 0; j < convex[i].size(); j++)
		{
			cv::Vec4i v = convex[i][j];
			int depth = v[3];
			cv::Point startPoint = contours[i][v[0]];
			cv::Point endPoint = contours[i][v[1]];
			cv::Point farPoint = contours[i][v[2]];

			if (depth > 1000) {
				cv::circle(handLayer1Depth, farPoint, 3, cv::Scalar(0, 0, 255), -1);
			}
			else {
				//cv::circle(handLayer1Depth, farPoint, 3, cv::Scalar(0, 255, 0), -1);
			}
		}
	}

	//cv::adaptiveThreshold(handLayer1Depth, handLayer1Depth, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 2);
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

void Application::buildEdgeMask()
{
	for (int y = 0; y < 480; y++) {
		ushort* rawRow = rawDepthFrame.ptr<ushort>(y);
		uchar* handMaskRow = handMask.ptr<uchar>(y);
		uchar* edgeMaskRow = edgeMask.ptr<uchar>(y);

		for (int x = 0; x < 640; x++) {
			int m = handMaskRow[x];
			int z = rawRow[x];
			if (z == 0 || m == 0) {
				edgeMaskRow[x] = 128;
				continue;
			}
			int cx, cy;
			kinectReader.convertDepthToColor(x, y, z, &cx, &cy);
			if (cx < 0 || cx >= 640 || cy < 0 || cy >= 480 || z == 0) {
				edgeMaskRow[x] = 128;
			}
			else {
				edgeMaskRow[x] = edgeColorFrame.at<uchar>(cy, cx);
			}
		}
	}
	cv::bitwise_not(edgeMask, edgeMask);
}

void Application::evaluateHandLayer1() // ~66ms
{
	/*cv::bitwise_and(handLayer1, edgeMask, handLayer1);
	cv::Mat openningKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(handLayer1, handLayer1, cv::MORPH_OPEN, openningKernel, cv::Point(-1, -1), 2);*/
	handLayer1Corners.clear();
	cv::Mat corner;
	cv::cornerHarris(handLayer1, corner, 8, 5, 0.04, cv::BORDER_DEFAULT);
	cv::normalize(corner, corner, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	for (int j = 0; j < corner.rows; j++) {
		float* cornerRow = corner.ptr<float>(j);
		for (int i = 0; i < corner.cols; i++) {
			if (cornerRow[i] > CORNER_THRESHOLD) {
				if (handLayer1.ptr<uchar>(j)[i] > 0) {
					handLayer1Corners.push_back(cv::Point(i, j));
				}
			}
		}
	}

	// group corner by contours
	vector<cv::Vec4i> hierachy;
	cv::findContours(handLayer1, contoursL1, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	// find all centroid
	vector<cv::Moments> mu(contoursL1.size());
	for (int i = 0; i < mu.size(); i++)
	{
		mu[i] = cv::moments(contoursL1[i], false);
	}
	vector<cv::Point2f> mc(contoursL1.size());
	for (int i = 0; i < contoursL1.size(); i++)
	{
		mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

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
	// cluster very near corner
	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		vector<cv::Point> cluster;
		clusterPoint(it->second, cluster, DISTANCE_THRESHOLD_CORNER_LAYER_1);
		cornerGroup[it->first] = cluster;
	}
	
	
	for (int i = 0; i < handLayer1Corners.size(); i++) {
		cv::circle(handLayer1, handLayer1Corners[i], 1, cv::Scalar(0, 0, 255), -1);
	}

	/*for (int i = 0; i < mc.size(); i++)
	{
		cv::circle(handLayer1, mc[i], 2, cv::Scalar(0, 255, 0), -1);
	}*/

	cv::circle(handLayer1, palmPoint, 4, cv::Scalar(0, 102, 255), -1);
}

void Application::evaluateHandLayer2()	// 7ms
{

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
			}
		}
	}
	
	clusterPoint(semi_fingerPoint, fingerL2Point, DISTANCE_THESHOLD);
	int count = 0;
	for (int i = 0; i < fingerL2Point.size(); i++) {
		cv::circle(handLayer2, fingerL2Point[i], 4, cv::Scalar(0, 255, 0), 2);
		count += 1;
	}
	if (count == 0)
		return;

	// find finger centroid point
	cv::Point fingerCentroid = calCentroid(fingerL2Point);
	cv::circle(handLayer2, fingerCentroid, 4, cv::Scalar(100, 255, 100), -1);
}

void Application::evaluateHandLayer3()
{
	vector<cv::Vec4i> hieracry;
	vector<vector<cv::Point>> contours;
	cv::findContours(handLayer3, contours, hieracry, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	
	if (contours.size() == 0)
		return;
	
	double largestArea = 0;
	int largestIndex = 0;
	for (int i = 0; i < contours.size(); i++) {
		double a = cv::contourArea(contours[i]);
		if (a > largestArea) {
			largestArea = a;
			largestIndex = i;
		}
	}

	vector<cv::Point> largestContour = contours[largestIndex];
	
	vector<int> largestHull;
	cv::convexHull(largestContour, largestHull);
	vector<cv::Vec4i> convexDefect;
	cv::convexityDefects(largestContour, largestHull, convexDefect);


	// find heightest contour point
	int heightestY = 480;
	int heightestIndex = -1;
	for (int i = 0; i < largestContour.size(); i++) {
		if (largestContour[i].y <= heightestY) {
			heightestY = largestContour[i].y;
			heightestIndex = i;
		}
	}
	if (heightestIndex == -1)
		return;
	cv::Point heightestPoint = largestContour[heightestIndex];
	cv::Rect boundingBox = cv::boundingRect(largestContour);
	cv::RotatedRect minAreaBox = cv::minAreaRect(largestContour);
	cv::Point2f vertices2f[4];
	minAreaBox.points(vertices2f);

	cv::Point boundingBoxTopMedian = calMedianPoint(cv::Point(boundingBox.x, boundingBox.y), cv::Point(boundingBox.x + boundingBox.width, boundingBox.y));

	cv::Vec2d boundingToHandLine = calLinear(boundingBoxTopMedian, handPoint);	// L
	cv::Point bhl_p1, bhl_p2;
	calEndpoint(boundingToHandLine, bhl_p1, bhl_p2);

	cv::Vec2d heightestToHandLine = calLinear(heightestPoint, handPoint);
	cv::Point hhl_p1, hhl_p2;
	calEndpoint(heightestToHandLine, hhl_p1, hhl_p2);

	double distTopHand = calDistance(handPoint, boundingBoxTopMedian);
	double distHeightestHand = calDistance(handPoint, heightestPoint);
	double distHandPalm = 2 * distTopHand - 80;
	double distHeightestPalm = 2 * distHeightestHand - 80;

	cv::Point p1_1, p1_2;
	calLinearInterceptCirclePoint(handPoint, distHandPalm, boundingToHandLine, p1_1, p1_2);
	cv::Point p4_1, p4_2;
	calLinearInterceptCirclePoint(handPoint, distHeightestPalm, heightestToHandLine, p4_1, p4_2);
	
	vector<cv::Point> convex;
	for (int i = 0; i < convexDefect.size(); i++) {
		cv::Vec4i v = convexDefect[i];
		int depth = v[3];
		if (depth < CONVEX_DEPTH_THRESHOLD_LAYER_3)
			continue;

		cv::Point farPoint = largestContour[v[2]];
		convex.push_back(farPoint);
		/*cv::circle(handLayer3, farPoint, 4, cv::Scalar(0, 102, 255), 2);*/
	}

	cv::cvtColor(handLayer3, handLayer3, cv::COLOR_GRAY2BGR);

	
	if (convex.size() <= 2) {
		// use height point
		double d1 = calDistance(heightestPoint, p4_1);
		double d2 = calDistance(heightestPoint, p4_2);

		/*if (d1 < d2) {
			palmPoint.x = p4_2.x;
			palmPoint.y = p4_2.y;
		}
		else {
			palmPoint.x = p4_1.x;
			palmPoint.y = p4_1.y;
		}*/
	}
	else {
		// use convex
		double maxConvexDist = 0;
		int edgeConvexIndex1 = 0;
		int edgeConvexIndex2 = 0;

		for (int i = 0; i < convex.size(); i++) {
			for (int j = i + 1; j < convex.size(); j++) {
				double d = calDistance(convex[i], convex[j]);
				if (d > maxConvexDist) {
					maxConvexDist = d;
					edgeConvexIndex1 = i;
					edgeConvexIndex2 = j;
				}
			}
		}
		cv::Point pe1 = convex[edgeConvexIndex1];
		cv::Point pe2 = convex[edgeConvexIndex2];

		
		vector<cv::Point> intercepts;
		for (int i = 0; i < convex.size(); i++) {
			if (i == edgeConvexIndex1 || i == edgeConvexIndex2) {
				cv::circle(handLayer3, convex[i], 4, cv::Scalar(0, 102, 255), -1);
				continue;
			}

			cv::circle(handLayer3, convex[i], 4, cv::Scalar(0, 102, 255), 2);
			cv::Point pi = convex[i];
			cv::Vec2d l1 = calLinear(pi, pe1);
			cv::Vec2d l2 = calLinear(pi, pe2);
			cv::Point mp1 = calMedianPoint(pi, pe1);
			cv::Point mp2 = calMedianPoint(pi, pe2);
			cv::Vec2d lp1 = calPerpendicularLine(l1, mp1);
			cv::Vec2d lp2 = calPerpendicularLine(l2, mp2);
			cv::Point intercept = calInterceptPoint(lp1, lp2);

			intercepts.push_back(intercept);
		}

		cv::Point center = calCentroid(intercepts);
		/*palmPoint.x = center.x;
		palmPoint.y = center.y;*/

		cv::circle(handLayer3, center, handRadius, cv::Scalar(0, 255, 0), 2);
		cv::circle(handLayer3, center, 4, cv::Scalar(0, 255, 0), -1);
	}

	cv::circle(handLayer3, handPoint, handRadius, cv::Scalar(255, 0, 0), 2);
	cv::circle(handLayer3, handPoint, 4, cv::Scalar(255, 0, 0), -1);
	/*cv::circle(handLayer3, palmPoint, 4, cv::Scalar(0, 102, 255), -1);
	cv::circle(handLayer3, palmPoint, handRadius, cv::Scalar(0, 102, 255), 2);*/
}

void Application::evaluateHandLayerPalm()
{
	handLayer3.copyTo(handLayerPalm);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//cv::morphologyEx(handLayerPalm, handLayerPalm, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
	vector<cv::Vec4i> hierachy;
	vector<vector<cv::Point>> contours;
	cv::findContours(handLayerPalm, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	
	if (contours.size() == 0)
		return;

	double largestArea = 0;
	int largestIndex = 0;
	for (int i = 0; i < contours.size(); i++) {
		double a = cv::contourArea(contours[i]);
		if (a > largestArea) {
			largestArea = a;
			largestIndex = i;
		}
	}

	vector<cv::Point> largestContour = contours[largestIndex];
	cv::drawContours(handLayerPalm, contours, largestIndex, cv::Scalar(255), -1);
	vector<int> largestHull;
	cv::convexHull(contours[largestIndex], largestHull);
	vector<cv::Vec4i> defect;
	cv::convexityDefects(contours[largestIndex], largestHull, defect);
	cv::Rect boundingBox = cv::boundingRect(largestContour);
	vector<cv::Point> concavePoints;
	for (int i = 0; i < defect.size(); i++) {
		cv::Vec4i v = defect[i];
		int depth = v[3];
		if (depth < 3000) {
			continue;
		}

		cv::Point farPoint = contours[largestIndex][v[2]];
		concavePoints.push_back(farPoint);
	}

	cv::Point minConcave;
	if (concavePoints.size() > 0) {
		cv::Mat hideThumbMask(handLayerPalm.size(), CV_8UC1, cv::Scalar(255));
		int minConcaveX = 640;
		minConcave = concavePoints[0];
		for (int i = 0; i < concavePoints.size(); i++)
		{
			if (concavePoints[i].x < minConcaveX) {
				minConcaveX = concavePoints[i].x;
				minConcave = concavePoints[i];
			}
		}

		cv::rectangle(hideThumbMask, cv::Rect(cv::Point(boundingBox.x, boundingBox.y), cv::Size(minConcave.x - boundingBox.x, boundingBox.height)), cv::Scalar(128), -1);
		cv::bitwise_and(hideThumbMask, handLayerPalm, handLayerPalm);
	}
	
	
	vector<cv::Vec2i> regions;
	vector<bool> changingTable(boundingBox.height+4, false);
	vector<bool> acceptVariationTable(boundingBox.height, false);
	vector<bool> acceptLengthTable(boundingBox.height, false);
	for (int y = boundingBox.y; y < boundingBox.y + boundingBox.height; y++) {
		uchar* handPalmRow = handLayerPalm.ptr<uchar>(y);
		int changing = 0;
		int prev = 0;
		int countWhite = 0;
		for (int x = boundingBox.x - 5; x < boundingBox.x + boundingBox.width + 5; x++) {
			int cur = handPalmRow[x];
			int rawCur = cur;
			int next = handPalmRow[x + 2];

			if (cur == 128) cur = 0;
			if (next == 128) next = 0;
			if (rawCur == 128) rawCur = 255;

			if (cur != prev && prev != next) {
				changing += 1;
				prev = cur;
			}

			if (rawCur == 255) {
				countWhite += 1;
			}
		}

		if (changing <= 2) {
			if (countWhite >= 1.5*handRadius) {
				cv::rectangle(handLayerPalm, cv::Rect(cv::Point(boundingBox.x + boundingBox.width + 10, y), cv::Size(10, 1)), cv::Scalar(255), -1);
				acceptLengthTable[y - boundingBox.y] = true;
			}
			else {
				cv::rectangle(handLayerPalm, cv::Rect(cv::Point(boundingBox.x + boundingBox.width + 10, y), cv::Size(10, 1)), cv::Scalar(128), -1);
			}
			changingTable[y - boundingBox.y] = true;
			acceptVariationTable[y - boundingBox.y] = true;
		}
	}

	int start_accept_length = -1;
	int maxRange = -1;
	int maxStart = -1;
	int maxEnd = -1;
	for (int i = 0; i < acceptLengthTable.size(); i++)
	{
		if (acceptLengthTable[i]) {
			if (start_accept_length == -1) {
				start_accept_length = i;
			}
		}
		else {
			if (start_accept_length != -1) {
				bool isMergeSection = false;
				for (int j = 1; j <= 10 && i + j < acceptLengthTable.size(); j++) {
					if (acceptLengthTable[i + j]) {
						isMergeSection = true;
						break;
					}
				}
				if (isMergeSection) {
					continue;
				}
				int range = i - start_accept_length;
				if (range > maxRange) {
					maxRange = range;
					maxStart = start_accept_length;
					maxEnd = i;
				}
				start_accept_length = -1;
			}
		}
	}
	if (start_accept_length != -1) {
		int range = acceptLengthTable.size() - start_accept_length;
		if (range > maxRange) {
			maxRange = range;
			maxStart = start_accept_length;
			maxEnd = acceptLengthTable.size();
		}
	}

	cv::Point center1, center2;
	bool hasCenter1 = false, hasCenter2 = false;
	if (maxRange != -1) {
		cv::rectangle(handLayerPalm, cv::Rect(cv::Point(boundingBox.x + boundingBox.width + 50, boundingBox.y + maxStart), cv::Size(10, maxRange)), cv::Scalar(255, 255, 255), -1);

		{
			cv::Mat mask(handLayerPalm.size(), CV_8UC1, cv::Scalar(0));
			cv::Mat sub;
			int x = boundingBox.x;
			int y = boundingBox.y + maxStart;
			int w = boundingBox.width;
			int h = maxRange;
			cv::Point center(x + w / 2, y + h / 2);
			cv::rectangle(mask, cv::Rect(cv::Point(x, y), cv::Size(w, h)), cv::Scalar(255), -1);
			cv::bitwise_and(handLayerPalm, mask, sub);
			cv::Moments m = cv::moments(sub, true);
			center1 = cv::Point(m.m10 / m.m00, m.m01 / m.m00);
			hasCenter1 = true;
			/*cv::rectangle(handLayerPalm, cv::Rect(cv::Point(x, y), cv::Size(w, h)), cv::Scalar(0, 0, 255), 1);
			cv::circle(handLayerPalm, center, 4, cv::Scalar(0, 0, 255), -1);
			cv::circle(handLayerPalm, center, handRadius, cv::Scalar(0, 0, 255), 1);*/
		}

		if (concavePoints.size() > 0) {
			cv::Mat mask(handLayerPalm.size(), CV_8UC1, cv::Scalar(0));
			cv::Mat sub;
			int x = minConcave.x;
			int y = boundingBox.y + maxStart;
			int w = boundingBox.x + boundingBox.width - minConcave.x;
			int h = maxRange;
			cv::Point center(x + w / 2, y + h / 2);
			cv::rectangle(mask, cv::Rect(cv::Point(x, y), cv::Size(w, h)), cv::Scalar(255), -1);
			cv::bitwise_and(handLayerPalm, mask, sub);

			cv::Moments m = cv::moments(sub, true);
			center2 = cv::Point(m.m10 / m.m00, m.m01 / m.m00);
			hasCenter2 = true;

			/*cv::rectangle(handLayerPalm, cv::Rect(cv::Point(x, y), cv::Size(w, h)), cv::Scalar(0, 0, 255), 1);
			cv::circle(handLayerPalm, center, 4, cv::Scalar(0, 0, 255), -1);
			cv::circle(handLayerPalm, center, handRadius, cv::Scalar(0, 0, 255), 1);*/
		}
	}

	
	int start_region = 0;
	bool regioning = false;
	bool prev = false;
	for (int i = 0; i < changingTable.size()-4; i++) {
		bool cur = changingTable[i];  
		bool next = changingTable[i + 4];

		if (cur != prev && prev != next) {
			prev = cur;
			if (regioning) {
				cv::rectangle(handLayerPalm, cv::Rect(cv::Point(boundingBox.x + boundingBox.width + 30, boundingBox.y + start_region), cv::Size(10, i - start_region)), cv::Scalar(128), -1);
			}
			else {
				start_region = i;
			}
			regioning = !regioning;
		}
	}
	if (regioning) {
		cv::rectangle(handLayerPalm, cv::Rect(cv::Point(boundingBox.x + boundingBox.width + 30, boundingBox.y + start_region), cv::Size(10, changingTable.size() - 4 - start_region)), cv::Scalar(128), -1);
	}

	cv::Mat palmCenter1Mask = cv::Mat::zeros(handLayerPalm.size(), CV_8UC1);
	cv::Mat palmCenter2Mask = cv::Mat::zeros(handLayerPalm.size(), CV_8UC1);
	cv::Mat palmTestCenter1, palmTestCenter2;
	if (hasCenter1) {
		cv::circle(palmCenter1Mask, center1, handRadius, cv::Scalar(255), -1);
		palmPoint.x = center1.x;
		palmPoint.y = center1.y;
		/*cv::circle(handLayerPalm, center1, 4, cv::Scalar(0, 0, 255), 1);
		cv::circle(handLayerPalm, center1, handRadius, cv::Scalar(0, 0, 255), 1);*/
	}
	if (hasCenter2) {
		/*cv::circle(handLayerPalm, center2, 4, cv::Scalar(0, 102, 255), 1);
		cv::circle(handLayerPalm, center2, handRadius, cv::Scalar(0, 102, 255), 1);*/
		cv::circle(palmCenter2Mask, center2, handRadius, cv::Scalar(255), -1);
		palmPoint.x = center2.x;
		palmPoint.y = center2.y;
	}

	cv::bitwise_and(handLayerPalm, palmCenter1Mask, palmTestCenter1);
	cv::bitwise_and(handLayerPalm, palmCenter2Mask, palmTestCenter2);
	int whiteCenter1 = cv::countNonZero(palmTestCenter1);
	int whiteCenter2 = cv::countNonZero(palmTestCenter2);

	cv::imshow("Palm Test C1", palmTestCenter1);
	cv::imshow("Palm Test C2", palmTestCenter2);

	cv::cvtColor(handLayerPalm, handLayerPalm, cv::COLOR_GRAY2BGR);
	cv::rectangle(handLayerPalm, boundingBox, cv::Scalar(0, 255, 0), 2);
	

	

	if (whiteCenter1 > whiteCenter2) {
		cv::circle(handLayerPalm, center1, 4, cv::Scalar(0, 0, 255), 1);
		cv::circle(handLayerPalm, center1, handRadius, cv::Scalar(0, 0, 255), 1);
		palmPoint.x = center1.x;
		palmPoint.y = center1.y;
	}
	else {
		cv::circle(handLayerPalm, center2, 4, cv::Scalar(0, 102, 255), 1);
		cv::circle(handLayerPalm, center2, handRadius, cv::Scalar(0, 102, 255), 1);
		palmPoint.x = center2.x;
		palmPoint.y = center2.y;
	}

	// find wrist point
	cv::Point wristPoint = cv::Point(0, 0);
	for (int i = 0; i < concavePoints.size(); i++)
	{
		if (concavePoints[i].x < palmPoint.x && concavePoints[i].y > wristPoint.y)
			wristPoint = concavePoints[i];
	}

	double wristAngle = calAnglePoint(palmPoint, wristPoint);

	for (int i = 0; i < concavePoints.size(); i++)
	{
		cv::Point concave = concavePoints[i];
		
		//cv::line(handLayerPalm, cv::Point(concave.x, boundingBox.y), cv::Point(concave.x, boundingBox.y + boundingBox.height), cv::Scalar(0, 102, 255), 2);
		cv::Point2d concavePolar = convertPointCartesianToPolar(concave, palmPoint);
		
		double theta = calAnglePoint(palmPoint, concave);

		if (theta > PI)
			cv::circle(handLayerPalm, concave, 6, cv::Scalar(0, 255, 255), 2);
		
		cout << "concave angle " << theta << ":" << theta-wristAngle << endl;

		int border = 2;
		if (abs(theta) > PI / 2)
			border = -1;
		if (theta < 0) {
			cv::circle(handLayerPalm, concave, 4, cv::Scalar(0, 0, 255), border);
		}
		else {
			cv::circle(handLayerPalm, concave, 4, cv::Scalar(0, 102, 255), border);
		}

		cv::Point endPoint1 = calRadiusPoint(theta, handRadius, palmPoint);
		cv::circle(handLayerPalm, wristPoint, 4, cv::Scalar(0, 255, 0), -1);
		//cv::line(handLayerPalm, palmPoint, endPoint1, cv::Scalar(0, 255, 0), 2);
	}

	double estimateAngle[] = {
		-3.0, -3.5, 2.4, 1.3, -0.7
	};
	for (int i = 0; i < 5; i++)
	{
		cv::Point endPoint = calRadiusPoint(estimateAngle[i] + wristAngle, handRadius, palmPoint);
		cv::line(handLayerPalm, palmPoint, endPoint, cv::Scalar(0, i * 50, 255), 2);
	}
	cv::line(handLayerPalm, palmPoint, wristPoint, cv::Scalar(0, 255, 0), 2);

	cv::circle(handLayerPalm, wristPoint, 4, cv::Scalar(0, 255, 0), -1);
}

void Application::evaluate3Layer()
{

	fingerPointL12.clear();
	//find layer 2 finger centroid
	cv::Point centroidL2 = calCentroid(fingerL2Point);

	vector<cv::Point> fingerL1Point, fingerL1PointCentroid, fingerL1PointMultiple, fingerL1PointPalm;
	vector<bool> fingerL2Used(fingerL2Point.size(), false);
	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++) {
		vector<cv::Point> group = it->second;
		vector<cv::Point> contourL1 = contoursL1[it->first];
		bool ignore = false;
		for (int i = 0; i < fingerL2Point.size(); i++) {
			double d = cv::pointPolygonTest(contourL1, fingerL2Point[i], true);
			if (d >= MIN_DIST_12) {
				cv::circle(handLayer2, fingerL2Point[i], 6, cv::Scalar(0, 0, 255), 2);
				cv::drawContours(handLayer2, contoursL1, it->first, cv::Scalar(0, 0, 255), 2);
				//group.push_back(fingerL2Point[i]);
				fingerL2Used[i] = true;
				fingerPointL12.push_back(fingerL2Point[i]);
				ignore = true;
				break;
			}
		}
		if (ignore)
			continue;

		for (int i = 0; i < group.size(); i++) {
			cv::Point corner = group[i];
			for (int j = 0; j < fingerL2Point.size(); j++) {
				cv::Point finger = fingerL2Point[j];
				double dist = calDistance(finger, corner);
				if (dist < DISTANCE_THESHOLD+5) {
					ignore = true;
				}
			}
		}

		if (ignore)
			continue;
		double farthestDist = 0;
		double farthestDistC = 0;
		double farthestDistMulti = 0;
		double farthestDistPalm = 0;
		double nearestDistPalm = 1000;
		int farthestIndex = 0;
		int farthestIndexC = 0;
		int farthestIndexM = 0;
		int farthestIndexP = 0;
		int nearestIndexP = 0;

		int inPalm = false;
		
		for (int i = 0; i < group.size(); i++) {
			double d = cv::pointPolygonTest(hullL2, group[i], true);
			double dc = calDistance(group[i], centroidL2);
			double dp = calDistance(group[i], palmPoint);
			if (d > farthestDist) {
				farthestDist = d;
				farthestIndex = i;
			}
			if (dc > farthestDistC) {
				farthestDistC = dc;
				farthestIndexC = i;
			}
			if (dc * d > farthestDistMulti) {
				farthestDistMulti = dc * d;
				farthestIndexM = i;
			}
			if (dp > farthestDistPalm) {
				farthestDistPalm = dp;
				farthestIndexP = i;
			}
			if (dp < nearestDistPalm) {
				nearestDistPalm = dp;
				nearestIndexP = i;
			}
			if (dp <= handRadius + 5) {
				inPalm = true;
			}
			
			cv::circle(handLayer2, group[i], 1, cv::Scalar(0, 255, 255), -1);

			if (dp > 1000)
				continue;
			char buffer[10];
			sprintf_s(buffer, "%.2f", dp);
			cv::putText(handLayer2, buffer, cv::Point(group[i].x, group[i].y + 10), cv::FONT_HERSHEY_COMPLEX, 0.2, cv::Scalar(0, 102, 255), 1);
			//cv::circle(handLayer2, group[i], abs(d), cv::Scalar(255, 128, 0), 2);
		}

		fingerL1PointCentroid.push_back(group[farthestIndexC]);
		fingerL1Point.push_back(group[farthestIndex]);
		fingerL1PointMultiple.push_back(group[farthestIndexM]);

		if (inPalm) {
			fingerL1PointPalm.push_back(group[nearestIndexP]);
		}
		else {
			fingerL1PointPalm.push_back(group[farthestIndexP]);
		}
		
	}

	if (hullL2.size() != 0)
		cv::drawContours(handLayer2, vector<vector<cv::Point>> {hullL2}, 0, cv::Scalar(0, 255, 0), 1);

	for (int i = 0; i < fingerL1PointPalm.size(); i++) {
		cv::circle(handLayer2, fingerL1PointPalm[i], 4, cv::Scalar(0, 0, 255), 2);

		fingerPointL12.push_back(fingerL1PointPalm[i]);
	}
	//if (fingerL2Point.size() > 0) {
	//	// use farthest from centroid
	//	for (int i = 0; i < fingerL1PointCentroid.size(); i++) {
	//		cv::circle(handLayer2, fingerL1PointCentroid[i], 4, cv::Scalar(255, 0, 0), 2);

	//		fingerPointL12.push_back(fingerL1PointCentroid[i]);
	//	}
	//}
	//else {
	//	// use farthest from hull
	//	/*for (int i = 0; i < fingerL1Point.size(); i++) {
	//		cv::circle(handLayer2, fingerL1Point[i], 4, cv::Scalar(0, 0, 255), 2);

	//		fingerPointL12.push_back(fingerL1Point[i]);
	//	}*/

	//	// use farthest from palm
	//	for (int i = 0; i < fingerL1PointPalm.size(); i++) {
	//		cv::circle(handLayer2, fingerL1PointPalm[i], 4, cv::Scalar(0, 0, 255), 2);

	//		fingerPointL12.push_back(fingerL1PointPalm[i]);
	//	}

	//}
	
	for (int i = 0; i < fingerL2Used.size(); i++) {
		if (!fingerL2Used[i])
			fingerPointL12.push_back(fingerL2Point[i]);
	}
	/*for (int i = 0; i < fingerL1Point.size(); i++) {
		cv::circle(handLayer2, fingerL1Point[i], 4, cv::Scalar(0, 0, 255), 2);
	}
	for (int i = 0; i < fingerL1PointCentroid.size(); i++) {
		cv::circle(handLayer2, fingerL1PointCentroid[i], 4, cv::Scalar(255, 0, 0), 2);
	} 
	for (int i = 0; i < fingerL1PointMultiple.size(); i++) {
		cv::circle(handLayer2, fingerL1PointMultiple[i], 4, cv::Scalar(255, 0, 255), 2);
	} */

	
	for (int i = 0; i < fingerPointL12.size(); i++) {
		cv::circle(handLayer3, fingerPointL12[i], 4, cv::Scalar(0, 0, 255), 2);
		cv::circle(handLayerAbs, fingerPointL12[i], 4, cv::Scalar(0, 0, 255), 2);
	}
	
	int handRadiusLess = 0.5 * handRadius;
	cv::circle(handLayer2, palmPoint, 4, cv::Scalar(0, 102, 255), -1);
	cv::circle(handLayer2, palmPoint, handRadius, cv::Scalar(0, 102, 255), 2);
	cv::circle(handLayer2, palmPoint, handRadiusLess, cv::Scalar(0, 102, 255), 2);

	cv::circle(handLayerAbs, palmPoint, 4, cv::Scalar(0, 102, 255), -1);
	cv::circle(handLayerAbs, palmPoint, handRadius, cv::Scalar(0, 102, 255), 2);
}

void Application::assignFingerId()
{
	vector<cv::Point2d> fingerPointL12Polar(fingerPointL12.size());
	vector<cv::Point3f> fingerPoint3d(fingerPointL12.size());
	FingerSorter fingerSorter;
	fingerSorter.origin = palmPoint;
	sort(fingerPointL12.begin(), fingerPointL12.end(), fingerSorter);
	for (int i = 0; i < fingerPointL12.size(); i++)
	{
		fingerPointL12Polar[i] = convertPointCartesianToPolar(fingerPointL12[i]);
		fingerPoint3d[i] = convertPoint2dTo3D(fingerPointL12[i]);
	}

	string fingerNames[] = {
		"Middle", "Ring", "Little", "Thumb", "Index"
	};
	int fingerIds[] = {
		FINGER_MIDDLE, FINGER_RING, FINGER_LITTLE, FINGER_THUMB, FINGER_INDEX
	};

	for (int i = 0; i < fingerPoint3d.size(); i++)
	{
		cv::Point3f p = fingerPoint3d[i];
		char buffer[100];
		sprintf_s(buffer, "(%.2f, %.2f, %.2f)", p.x, p.y, p.z);
		printf("(%.2f, %.2f, %.2f)\n", p.x, p.y, p.z);
		cv::putText(handLayerAbs, buffer, cv::Point(fingerPointL12[i].x, fingerPointL12[i].y + 10), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
	}
	
	if (palmPoint.x == 0 || palmPoint.y == 0)
		return;

	palmPoint3d = convertPoint2dTo3D(palmPoint);
	char buffer[100];
	sprintf_s(buffer, "(%.2f, %.2f, %.2f)", palmPoint3d.x, palmPoint3d.y, palmPoint3d.z);
	cv::putText(handLayerAbs, buffer, cv::Point(palmPoint.x, palmPoint.y + 10), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 102, 255), 1);
	for (int i = 0; i < fingerPointL12.size() && i < 5; i++)
	{
		finger3dMap[fingerIds[i]] = fingerPoint3d[i];
		finger3ds[fingerIds[i]] = fingerPoint3d[i];
		/*char buffer[10];
		sprintf_s(buffer, "%s", fingerNames[i].c_str());
		cv::putText(handLayerAbs, buffer, cv::Point(fingerPointL12[i].x, fingerPointL12[i].y + 10), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 102, 255), 1);*/
	}
	finger3dMap[PALM_POSITION] = palmPoint3d;
	finger3ds[PALM_POSITION] = palmPoint3d;

	for (int i = 0; i < 6; i++)
	{
		cv::Point3f p = finger3dMap[i];
		cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << endl;
	}
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

cv::Vec2d Application::calLinear(cv::Point p1, cv::Point p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	double m, c;
	if (dx == 0) {
		m = HUGE_VAL;
		c = p1.x;
	}
	else {
		m = dy / dx;
		c = p1.y - m * p1.x;
	}
	
	return cv::Vec2d(m, c);
}

cv::Point Application::calInterceptPoint(cv::Vec2d l1, cv::Vec2d l2)
{
	double m1 = l1[0];
	double c1 = l1[1];
	double m2 = l2[0];
	double c2 = l2[1];
	int x, y;

	/* paralell never cross together */
	//assert(m1 != m2);
	if (m1 == m2) {
		return cv::Point(INT_MAX, INT_MAX);
	}

	if (m1 == HUGE_VAL) {
		x = -c1;
		y = m2 * (-c1) + c2;
	}
	else if (m2 == HUGE_VAL) {
		x = -c2;
		y = m1 * (-c2) + c1;
	}
	else if (m1 == 0) {
		x = (c1 - c2) / m2;
		y = c1;
	}
	else if (m2 == 0) {
		x = (c2 - c1) / m1;
		y = c2;
	}
	else {
		x = (c2 - c1) / (m1 - m2);
		y = m1 * x + c1;
	}
	return cv::Point(x, y);
}

cv::Vec2d Application::calPerpendicularLine(cv::Vec2d l, cv::Point p)
{
	double m = l[0];
	double c = l[1];

	if (m == HUGE_VAL) {
		//assert(p.x == -c);
		return cv::Vec2d(0, p.y);
	}
	else if (m == 0) {
		//assert(p.y == c);
		return cv::Vec2d(HUGE_VAL, -p.x);
	}
	else {
		//assert(p.y == (int) (m * p.x + c));
		double m2 = -1 / m;
		double c2 = p.y + (p.x / m);
		return cv::Vec2d(m2, c2);
	}
}

void Application::calEndpoint(cv::Vec2d l, cv::Point & p1, cv::Point & p2)
{
	double m = l[0];
	double c = l[1];
	if (m == HUGE_VAL) {
		p1.x = -c;
		p1.y = 0;
		p2.x = -c;
		p2.y = 480;
	}
	else {
		p1.x = 0;
		p1.y = c;
		p2.x = 640;
		p2.y = 640 * m + c;
	}
}

cv::Point Application::calMedianPoint(cv::Point p1, cv::Point p2)
{
	int x = (p1.x + p2.x) / 2;
	int y = (p1.y + p2.y) / 2;
	return cv::Point(x, y);
}

cv::Point Application::calRatioPoint(cv::Point p1, cv::Point p2, double ratio1, double ratio2)
{
	double px = (ratio2 * p1.x + ratio1 * p2.x) / (ratio1 + ratio2);
	double py = (ratio2 * p1.x + ratio1 * p2.x) / (ratio1 + ratio2);

	return cv::Point(px, py);
}

cv::Point Application::calLinearPointByX(cv::Vec2d L, double x)
{
	double m = L[0];
	double c = L[1];
	double y = 0;
	if (m == 0) {
		y = c;
	}
	else if (m == HUGE_VAL) {
		y = 0;
	}
	else {
		y = m * x + c;
	}
	return cv::Point(x, y);
}

cv::Point Application::calLinearPointByY(cv::Vec2d L, double y)
{
	double m = L[0];
	double c = L[1];
	double x = 0;
	if (m == 0) {
		x = 0;
	}
	else if (m == HUGE_VAL) {
		x = -c;
	}
	else {
		x = y / m - c;
	}
	return cv::Point(x, y);
}

void Application::calLinearInterceptCirclePoint(cv::Point center, double r, cv::Vec2d linear, cv::Point & p_out1, cv::Point & p_out2)
{
	double h = center.x;
	double k = center.y;
	double m = linear[0];
	double c = linear[1];

	double a = (m * m) + 1;
	double b = (2 * m*(c - k)) - 2 * h;
	double c_2 = ((h*h) + (c - k)*(c - k) - (r*r));

	double p1_x1 = ((-1 * b) + sqrt(b*b - (4 * a*c_2))) / (2 * a);
	double p1_x2 = ((-1 * b) - sqrt(b*b - (4 * a*c_2))) / (2 * a);

	cv::Point p1_1 = calLinearPointByX(linear, p1_x1);
	cv::Point p1_2 = calLinearPointByX(linear, p1_x2);

	p_out1.x = p1_1.x;
	p_out1.y = p1_1.y;
	p_out2.x = p1_2.x;
	p_out2.y = p1_2.y;
}

cv::Point2d Application::convertPointCartesianToPolar(cv::Point p, cv::Point o)
{
	double r = sqrt(pow(p.x - p.x, 2) + pow(p.y - o.y, 2));
	double t = atan(((double)p.y - o.y) / ((double)p.x - o.x));
	return cv::Point2d(r, t);
}

cv::Point3f Application::convertPoint2dTo3D(cv::Point p)
{
	int x=0, y=0, z=0;
	//find min z
	for (int j = p.y-1; j <= p.y + 1; j++)
	{
		if (j < 0 || j >= 480)
			continue;
		for (int i = p.x-1; i <= p.x+1; i++)
		{
			if (i < 0 || i >= 640)
				continue;
			int zij = rawDepthFrame.at<ushort>(j, i);
			if ((zij != 0 && zij < z) || z == 0) {
				x = i;
				y = j;
				z = zij;
			}
		}
	}
	float wx, wy, wz;
	kinectReader.convertDepthToWorld(x, y, z, &wx, &wy, &wz);
	return cv::Point3f(wx, wy, wz);
}

cv::Point Application::calRadiusPoint(double angle, double radius, cv::Point origin)
{
	double k = radius * sin(angle);
	double h = sqrt(pow(radius, 2) - pow(k, 2));
	
	int y = origin.y + k;
	int x;
	if (angle > PI / 2 || angle < -PI / 2)
		x = origin.x - h;
	else
		x = origin.x + h;
	
	return cv::Point(x, y);
}

double Application::calAnglePoint(cv::Point origin, cv::Point p)
{
	double theta = 0;
	double h = (p.x - origin.x);
	double k = (p.y - origin.y);
	//theta = asin(k / sqrt(pow(k, 2) + pow(h, 2)));
	if (h >= 0) {
		if (h == 0 && k == 0)
			theta = 0;
		else
			theta = asin(k / sqrt(pow(k, 2) + pow(h, 2)));
	}
	else {
		theta = PI - asin(k / sqrt(pow(k, 2) + pow(h, 2)));
	}
	return theta;
}

void Application::sendData()
{
	adapterCaller.sendData(finger3ds);
}

void Application::captureFrame()
{
	time_t ts = time(nullptr);

	/*cv::Mat scaledHistogram, concat, rawHand, handMask16U, edgeFrameC3;
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
	cv::imwrite(buffer_concat, concat);*/
	char buffer_1[80], buffer_2[80], buffer_3[80], buffer_4[80];
	sprintf_s(buffer_1, "%d - HL1_COR.jpg", ts);
	sprintf_s(buffer_2, "%d - HL1_CON.jpg", ts);
	sprintf_s(buffer_3, "%d - HL1_COR_G.jpg", ts);
	sprintf_s(buffer_4, "%d - HL1.jpg", ts);
	
	//cv::imwrite(buffer_1, HL1_COR);

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
