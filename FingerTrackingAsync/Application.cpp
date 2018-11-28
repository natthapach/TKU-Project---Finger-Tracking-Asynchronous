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
		//kinectReader.readRGBFrame();
		kinectReader.readDepthFrame();
		//colorFrame = kinectReader.getRGBFrame();
		depthFrame = kinectReader.getDepthFrame();
		rawDepthFrame = kinectReader.getRawDepthFrame();
		
		if (kinectReader.isHandTracking()) {

			handPoint = kinectReader.getHandPoint();
			handRadius = kinectReader.getHandRadius(HAND_RADIUS_MM);
			
			buildDepthHandMask();
			buildHand3Layers();

			bool skip = false;
			cv::Mat substract, substract_L1;
			cv::bitwise_xor(handMask, prevHandMasks[2], substract);
			cv::bitwise_xor(handLayer1, prevHandLayer1[2], substract_L1);
			cv::erode(substract, substract, cv::Mat());
			cv::erode(substract_L1, substract_L1, cv::Mat());
			int wh = cv::countNonZero(substract);
 			int wh_L1 = cv::countNonZero(substract_L1);
			cout << "white " << wh << ", " << wh_L1 << endl;
			skip = wh < 30 || wh_L1 < 15;

			for (int i = 2; i > 0; i--)
			{
				prevHandMasks[i - 1].copyTo(prevHandMasks[i]);
				prevHandLayer1[i - 1].copyTo(prevHandLayer1[i]);
			}
			prevHandMasks[0] = handMask;
			prevHandLayer1[0] = handLayer1;

			if (!skip)
			{
				evaluateHandLayer3();
				evaluateHandLayerCut();
				evaluateHandLayer2();
				evaluateHandLayer1();
				evaluate3Layer();
				evaluatePalmAngle();

				assignFingerId();

				cv::imshow(WINDOW_MASK_L1, handLayer1); // 14
				cv::imshow(WINDOW_MASK_L2, handLayer2);
				cv::imshow(WINDOW_MASK_L3, handLayer3);
				cv::imshow("Hand Absolute", handLayerAbs);
			}

			displayResult();
			displayLabel();

			

			thread(&Application::sendData, this).detach();

			cv::imshow("substact", substract);
			
			
			cv::imshow("Hand Absolute 2", handLayerAbs2);
			//cv::imshow("Edge", edgeColorFrame);
			cv::imshow("Palm", handLayerPalm);
			cv::imshow("Cut", handLayerCut);
			cv::imshow("Cut Mask", cutMask);
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
	cv::circle(handMask, cv::Point(handPosX, handPosY), 4 * handRadius, cv::Scalar(0), 2);
	cv::floodFill(handMask, cv::Point((int)handPosX, (int)handPosY), cv::Scalar(255));
	cv::threshold(handMask, handMask, 129, 255, cv::THRESH_BINARY);
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


void Application::evaluateHandLayer1() // ~66ms
{
	if (extendedFinger.size() < 3) {
		cv::bitwise_and(handLayer1, cutMask, handLayer1);
	}
	vector<cv::Point> handLayer1Corners;
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
	vector<vector<cv::Point>> hulls(contoursL1.size());
	for (int i = 0; i < contoursL1.size(); i++)
	{
		cv::convexHull(contoursL1[i], hulls[i]);
	}

	// vertex of contour
	vector<vector<cv::Point>> vertexes(hulls.size(), vector<cv::Point>());
	for (int i = 0; i < hulls.size(); i++)
	{
		for (int j = 0; j < hulls[i].size(); j++)
		{
			cv::Point start = hulls[i][(j - 1) % hulls[i].size()];
			cv::Point farP = hulls[i][j];
			cv::Point end = hulls[i][(j + 1) % hulls[i].size()];
			
			double angle = calAngleOf3Points(start, farP, end);
			if (angle < 0.5*PI)
			{
				vertexes[i].push_back(farP);
			}
		}
		
	}

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
			cv::drawContours(handLayer1, contoursL1, it->first, cv::Scalar(0, 0, 255), -1);
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
	
	for (int i = 0; i < vertexes.size(); i++)
	{
		for (int j = 0; j < vertexes[i].size(); j++)
		{
			cv::circle(handLayer1, vertexes[i][j], 2, cv::Scalar(0, 255, 0), -1);
			//cornerGroup[i].push_back(vertexes[i][j]);
		}
	}
}

void Application::evaluateHandLayer2()	// 7ms
{

	vector<cv::Point> largestContour = findLargestContour(handLayer2);
	if (largestContour.size() == 0) return;

	vector<int> hull;
	cv::convexHull(largestContour, hull);

	vector<cv::Point> semi_fingerPoint, abyss_finger;
	vector<cv::Vec4i> defect;
	cv::convexityDefects(largestContour, hull, defect);

	for (int i = 0; i < defect.size(); i++) {
		cv::Vec4i v = defect[i];
		int depth = v[3];
		if (depth > CONVEX_DEPTH_THRESHOLD_LAYER_2) {
			cv::Point startPoint = largestContour[v[0]];
			cv::Point endPoint = largestContour[v[1]];
			cv::Point farPoint = largestContour[v[2]];

			double ms = ((double)(startPoint.y - farPoint.y)) / (startPoint.x - farPoint.x);
			double me = ((double)(endPoint.y - farPoint.y)) / (endPoint.x - farPoint.x);
			double angle = atan((me - ms) / (1 + (ms * me))) * (180 / PI);

			if (angle < 0) {
				semi_fingerPoint.push_back(startPoint);
				semi_fingerPoint.push_back(endPoint);
			}
		}
	}
	
	clusterPoint(semi_fingerPoint, extendedFinger, DISTANCE_THESHOLD);

	if (extendedFinger.size() < 3) {
		cv::Mat addRegionMask = cv::Mat::zeros(handLayer1.size(), CV_8UC1);
		cv::Rect modifyRect;
		modifyRect.x = palmRect.x;
		modifyRect.y = 0;
		modifyRect.height = palmRect.y;
		modifyRect.width = palmRect.width;

		cv::rectangle(addRegionMask, modifyRect, cv::Scalar(255), -1);
		cv::bitwise_or(handLayer1, handLayer2, handLayer1, addRegionMask);
	}

	cv::cvtColor(handLayer2, handLayer2, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < extendedFinger.size(); i++) {
		cv::circle(handLayer2, extendedFinger[i], 4, cv::Scalar(0, 255, 0), 2);
	}
}

void Application::evaluateHandLayer3()
{
	// find largest contour and fill on
	vector<cv::Point> largestContour = findLargestContour(handLayer3);
	if (largestContour.size() == 0) return;
	cv::drawContours(handLayer3, vector<vector<cv::Point>> {largestContour}, 0, cv::Scalar(255), -1);
	
	// find bounding box to define resion
	cv::Rect boundingBox = cv::boundingRect(largestContour);

	// find concave points
	vector<int> largestHull;
	cv::convexHull(largestContour, largestHull);
	vector<cv::Point> concavePoints = findConcavePoints(largestContour, largestHull, 3000);
	vector<cv::Point> concavePointsLow = findConcavePoints(largestContour, largestHull, 1000);
	cv::Point minConcave;
	if (concavePoints.size() > 0) {
		cv::Mat hideThumbMask(handLayerPalm.size(), CV_8UC1, cv::Scalar(255));
		minConcave = EL3_findMinConcave(concavePoints);

		cv::rectangle(hideThumbMask, cv::Rect(cv::Point(boundingBox.x, boundingBox.y), cv::Size(minConcave.x - boundingBox.x, boundingBox.height)), cv::Scalar(128), -1);
		cv::bitwise_and(hideThumbMask, handLayer3, handLayer3);
	}

	vector<bool> acceptTransitionTable(boundingBox.height, false);
	vector<bool> acceptLengthTable(boundingBox.height, false);
	EL3_buildTable(handLayer3, boundingBox, acceptTransitionTable, acceptLengthTable);

	cv::Vec2i maxRegion = EL3_findMaxRegion(acceptTransitionTable, acceptLengthTable);
	
	palmRect = cv::Rect(cv::Point(boundingBox.x, boundingBox.y + maxRegion[0]), cv::Size(boundingBox.width, maxRegion[1] - maxRegion[0]));

	cv::Point center1 = EL3_findRegionCenter(handLayer3, palmRect);
	cv::Point center2;
	int white1 = 0, white2 = 0;
	white1 = EL3_countWhitePoint(handLayer3, center1, handRadius);
	if (concavePoints.size() > 0) {
		center2 = EL3_findRegionCenter(handLayer3, cv::Rect(cv::Point(minConcave.x, palmRect.y), cv::Size(palmRect.width - (minConcave.x-palmRect.x), palmRect.height)));
		white2 = EL3_countWhitePoint(handLayer3, center2, handRadius);
	}
	
	if (white1 > white2)
	{
		palmPoint.x = center1.x;
		palmPoint.y = center1.y;
	}
	else {
		palmPoint.x = center2.x;
		palmPoint.y = center2.y;
	}
	cv::bitwise_or(handLayer3, handMask, handLayer3);
	handLayer3.copyTo(handLayer3_2);


	// ---- find middle line ----
	cv::cvtColor(handLayer3, handLayer3, cv::COLOR_GRAY2BGR);
	cv::Point wristPoint1 = cv::Point(640, 0);
	for (int i = 0; i < concavePointsLow.size(); i++)
	{
		cv::Point ci = concavePointsLow[i];
		double d = calDistance(ci, palmPoint);
		if (ci.x < palmPoint.x && ci.y > wristPoint1.y && d <= 1.6*handRadius)
		{
			wristPoint1 = ci;
		}
	}
	cv::circle(handLayer3, wristPoint1, 4, cv::Scalar(0, 0, 255), -1);

	double wristDist = calDistance(wristPoint1, palmPoint);
	//cv::circle(handLayer3, palmPoint, wristDist, cv::Scalar(0, 255, 102), 2);

	vector<cv::Point> semi_wrist2;
	vector<cv::Point> semi_arm;
	for (int i = 0; i < largestContour.size(); i++)
	{
		double d = calDistance(largestContour[i], palmPoint);
		if (abs(d-wristDist) <= 2 ) {
			semi_wrist2.push_back(largestContour[i]);
		}
		if (abs(d - 2 * handRadius) <= 2) {
			semi_arm.push_back(largestContour[i]);
			
		}
		//cv::circle(handLayer3, largestContour[i], 2, cv::Scalar(255, 0, 0), -1);
	}
	// find arm point
	vector<cv::Point> semi_arm_cluster;
	clusterPoint(semi_arm, semi_arm_cluster, 5);
	cv::Point arm1 = cv::Point(0, 0), arm2 = cv::Point(0, 0);
	bool arm1_flag = false, arm2_flag = false;
	for (int i = 0; i < semi_arm_cluster.size(); i++)
	{
		cv::Point semi = semi_arm_cluster[i];
		if (semi.y < palmPoint.y)
		{
			continue;
		}
		if (semi.y > arm2.y)
		{
			arm2 = semi;
			arm2_flag = true;
		}
		if (semi.y > arm1.y)
		{
			arm2 = arm1;
			arm1 = semi;
			arm1_flag = true;
		}
		cv::circle(handLayer3, semi_arm_cluster[i], 2, cv::Scalar(255, 0, 0), -1);
	}
	cv::circle(handLayer3, arm1, 4, cv::Scalar(0, 102, 255), -1);
	cv::circle(handLayer3, arm2, 4, cv::Scalar(0, 102, 255), -1);


	cv::Point wristPoint2 = cv::Point(0, 0);
	for (int i = 0; i < semi_wrist2.size(); i++)
	{
		if (semi_wrist2[i].x > palmPoint.x && semi_wrist2[i].y > wristPoint2.y)
		{
			wristPoint2 = semi_wrist2[i];
		}
		cv::circle(handLayer3, semi_wrist2[i], 2, cv::Scalar(255, 255, 0), -1);
	}
	cv::circle(handLayer3, wristPoint2, 4, cv::Scalar(0, 102, 255), -1);

	cv::Point median_wrist = calMedianPoint(wristPoint1, wristPoint2);
	cv::Vec2d wristPalmLine = calLinear(median_wrist, palmPoint);
	cv::Point wep1, wep2;
	calEndpoint(wristPalmLine, wep1, wep2);
	cv::line(handLayer3, wep1, wep2, cv::Scalar(0, 0, 255), 2);
	cv::line(handLayerAbs, wep1, wep2, cv::Scalar(0, 255, 0), 2);
	cv::circle(handLayerAbs, wristPoint1, 4, cv::Scalar(255, 255, 0), -1);
	cv::circle(handLayerAbs, wristPoint2, 4, cv::Scalar(255, 255, 0), -1);
	cv::circle(handLayerAbs, palmPoint, 2*handRadius, cv::Scalar(255, 255, 0), 2);
	middleLine = wristPalmLine;

	
}

void Application::evaluateHandLayerCut()
{
	cutMask = cv::Mat(cv::Size(640, 480), CV_8UC1, cv::Scalar(255));
	cv::Mat sub, handLayer3_inverse;
	cv::Mat palmMask = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	handLayer3_2.copyTo(sub);

	cv::Rect modifyPalmRect = cv::Rect(cv::Point(palmRect.x, palmRect.y), cv::Size(palmRect.width + 10, palmRect.height));
	//cv::Rect modifyPalmRect = cv::Rect(cv::Point(palmRect.x, palmPoint.y - 10), cv::Size(palmRect.width + 10, 20));
	cv::rectangle(palmMask, modifyPalmRect, cv::Scalar(255), -1);
	cv::bitwise_not(handLayer3_2, handLayer3_inverse);
	cv::bitwise_and(handLayer3_inverse, palmMask, sub);

	cv::imshow("cut sub", sub);
	
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierachy;
	cv::findContours(sub, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	vector<vector<cv::Point>> hulls(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		cv::convexHull(contours[i], hulls[i]);
	}

	vector<cv::Point> hullCentroids(contours.size());
	for (int i = 0; i < hulls.size(); i++)
	{
		hullCentroids[i] = calCentroid(hulls[i]);
	}

	int rigthestX = 0;
	int rigthestIndex = -1;
	for (int i = 0; i < hullCentroids.size(); i++)
	{
		if (hullCentroids[i].x > rigthestX)
		{
			rigthestX = hullCentroids[i].x;
			rigthestIndex = i;
		}
	}

	if (rigthestIndex == -1)
		return;

	vector<cv::Point> rigthestHull = hulls[rigthestIndex];

	cv::Point endpoint_top = cv::Point(640, 480);
	cv::Point endpoint_buttom = cv::Point(640, 0);
	for (int i = 0; i < rigthestHull.size(); i++)
	{
		cv::Point hp = rigthestHull[i];
		if (hp.y < endpoint_top.y || (hp.y == endpoint_top.y && hp.x < endpoint_top.x))
			endpoint_top = hp;
		if (hp.y > endpoint_buttom.y || (hp.y == endpoint_buttom.y && hp.x < endpoint_buttom.x))
			endpoint_buttom = hp;
	}

	handDirection = calLinear(endpoint_top, endpoint_buttom);
	cv::Point ep1, ep2;
	calEndpoint(handDirection, ep1, ep2);
	double directionAngle = calLinerAngleByPoint(handDirection, palmPoint);
	cv::Point pr = calRadiusPoint(directionAngle, handRadius, palmPoint);


	double concave_predict[] = {
		-0.4,	// index-middle
		0.1,	// middle-ring
		0.5		// ring-little
	};

	vector<cv::Point> predictConcaves(3, cv::Point(0, 0));
	vector<cv::Vec2d> parallelLines(3);
	for (int i = 0; i < 3; i++)
	{
		predictConcaves[i] = calRadiusPoint(directionAngle + concave_predict[i], handRadius, palmPoint);
		parallelLines[i] = calParalellLine(handDirection, predictConcaves[i]);
	}

	handLayer3_2.copyTo(handLayerCut);
	cv::cvtColor(handLayerCut, handLayerCut, cv::COLOR_GRAY2BGR);
	
	cv::circle(handLayerCut, endpoint_top, 4, cv::Scalar(0, 102, 255), -1);
	cv::circle(handLayerCut, endpoint_buttom , 4, cv::Scalar(0, 102, 255), -1);

	cv::circle(handLayerCut, palmPoint, 4, cv::Scalar(0, 102, 255), -1);
	cv::circle(handLayerCut, palmPoint, handRadius, cv::Scalar(0, 102, 255), 2);
	cv::circle(handLayerCut, pr, 2, cv::Scalar(0, 0, 255), -1);

	for (int i = 0; i < 3; i++)
	{
		cv::Point ppl1, ppl2;
		calEndpoint(parallelLines[i], ppl1, ppl2);
		cv::line(handLayerCut, ppl1, ppl2, cv::Scalar(0, 255, 0), 1);

		//cv::line(handLayer1, ppl1, ppl2, cv::Scalar(0), 2);
		cv::line(cutMask, ppl1, ppl2, cv::Scalar(0), 2);
		cv::circle(handLayerCut, predictConcaves[i], 4, cv::Scalar(0, 102, 255), -1);
	}

	cv::line(handLayerCut, ep1, ep2, cv::Scalar(0, 0, 255), 2);
}

void Application::evaluate3Layer()
{

	fingerPoints.clear();

	// merge extended finger to corner group
	vector<bool> fingerL2Used(extendedFinger.size(), false);
	vector<int> countMerge(contoursL1.size(), 0);
	map<int, vector<cv::Point>> mergedFingerMap;
	vector<cv::Scalar> colors = {
		cv::Scalar(0, 0, 255),
		cv::Scalar(255, 0, 0),
		cv::Scalar(0, 255, 0),
		cv::Scalar(0, 102, 255),
		cv::Scalar(255, 0, 255)
	};
	for (map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++)
	{
		vector<cv::Point> corners = it->second;
		vector<cv::Point> contour = contoursL1[it->first];
		bool merge_flag = false;
		for (int i = 0; i < extendedFinger.size(); i++)
		{
			cv::Point finger = extendedFinger[i];
			double d = cv::pointPolygonTest(contour, finger, true);
			if (d > MIN_DIST_12) {
				it->second.push_back(finger);
				cv::circle(handLayer2, finger, 4, colors[it->first%5], 2);
				cv::drawContours(handLayer2, vector<vector<cv::Point>>{ contour }, 0, colors[it->first % 5], 2);
				fingerL2Used[i] = true;
				countMerge[it->first] += 1;
				merge_flag = true;
				
				if (mergedFingerMap.count(it->first) == 0)
				{
					mergedFingerMap[it->first] = vector<cv::Point>();
				}
				mergedFingerMap[it->first].push_back(finger);
			}
		}
		if (merge_flag)
		{
			clusterPoint(corners, corners, DISTANCE_THESHOLD);
		}
	}

	// collect unmerge extend finger
	for (int i = 0; i < fingerL2Used.size(); i++)
	{
		if (!fingerL2Used[i]) {
			fingerPoints.push_back(extendedFinger[i]);
		}
	}


	// select corner
	double handAngle = calLinerAngleByPoint(handDirection, palmPoint);
	double angle_low = handAngle - 0.75*PI + 0.1;
	double angle_up = handAngle + 0.75*PI;

	vector<cv::Point> selectedCorners;
	for(map<int, vector<cv::Point>>::iterator it = cornerGroup.begin(); it != cornerGroup.end(); it++)
	{
		// checking amount of corner to select
		vector<cv::Point> corners = it->second;
		int mergedFinger = countMerge[it->first];
		int select_amt = 0;
		if (mergedFinger < 2)
			select_amt = 1;
		else
			select_amt = mergedFinger;

		bool extendPalm = false;

		// making distance list
		vector<PointDistance> distances(corners.size());
		for (int i = 0; i < corners.size(); i++)
		{
			// filter out point in lower palm
			double angle = calAnglePoint(palmPoint, corners[i]);
			if (angle < angle_low || angle > angle_up)
				continue;

			double d = calDistance(palmPoint, corners[i]);
			distances[i].point = corners[i];
			distances[i].distance = d;
			if (d > 1.4*handRadius)
				extendPalm = true;

			cv::circle(handLayer2, corners[i], 1, cv::Scalar(0, 102, 255), -1);
		}

		// sort by distance
		PointDistanceSorter sorter;
		sort(distances.begin(), distances.end(), sorter);

		// filter out near point
		int i = 0;
		while (i < distances.size() - 1) {
			cv::Point pi = distances[i].point;
			cv::Point pj = distances[i + 1].point;
			double dij = calDistance(pi, pj);
			if (dij < DISTANCE_THESHOLD)
			{
				if (distances[i].distance > distances[i + 1].distance)
				{
					distances.erase(distances.begin() + i + 1);
					i += 2;
					
					continue;
				}
				else {
					distances.erase(distances.begin() + i);
					continue;
				}
			}
			i++;
		}

		if (extendPalm)
		{
			// use farthest
			if (mergedFinger > 0)
			{
				for (int i = 0; i < mergedFingerMap[it->first].size(); i++)
				{
					selectedCorners.push_back(mergedFingerMap[it->first][i]);
				}
			}
			else {
				for (int i = 0; i < distances.size() && i < select_amt; i++)
				{
					selectedCorners.push_back(distances[i].point);
				}
			}
		}
		else {
			for (int i = 0; i < distances.size() && i < select_amt; i++)
			{
				selectedCorners.push_back(distances[distances.size() - i - 1].point);
			}
		}
	}

	for (int i = 0; i < selectedCorners.size(); i++)
	{
		fingerPoints.push_back(selectedCorners[i]);
	}

	// filter out near finger point (redundant point from other contour group)
	int i = 0;
	while (i < (fingerPoints.size()-1) && fingerPoints.size() != 0) {
		cv::Point pi = fingerPoints[i];
		cv::Point pj = fingerPoints[i+1];
		double ai = calAnglePoint(palmPoint, pi);
		double aj = calAnglePoint(palmPoint, pj);
		double dij = calDistance(pi, pj);

		if (abs(ai-aj) < ASSIGN_FINGER_ANGLE_THRESHOLD && dij < DISTANCE_THESHOLD)
		{
			double di = calDistance(palmPoint, pi);
			double dj = calDistance(palmPoint, pj);
			if (di > dj)
			{
				fingerPoints.erase(fingerPoints.begin() + i + 1);
				i += 2;
				continue;
			}
			else {
				fingerPoints.erase(fingerPoints.begin() + i);
				continue;
			}
		}
		i++;
	}

	for (int i = 0; i < fingerPoints.size(); i++)
	{
		cv::circle(handLayerAbs, fingerPoints[i], 4, cv::Scalar(0, 0, 255), -1);  
	}
	
	cv::Point pl = calRadiusPoint(angle_low, handRadius, palmPoint);
	cv::Point pu = calRadiusPoint(angle_up, handRadius, palmPoint);

	cv::line(handLayerAbs, palmPoint, pl, RING_COLOR, 1);
	cv::line(handLayerAbs, palmPoint, pu, RING_COLOR, 1);
	
	cv::circle(handLayerAbs, palmPoint, 4, cv::Scalar(0, 102, 255), -1);
	cv::circle(handLayerAbs, palmPoint, handRadius, cv::Scalar(0, 102, 255), 2);
	cv::circle(handLayerAbs, palmPoint, 0.7*handRadius, cv::Scalar(0, 0, 255), 2);
	cv::circle(handLayerAbs, palmPoint, 1.5*handRadius, cv::Scalar(0, 102, 255), 2);
}

void Application::evaluatePalmAngle()
{
	double z_angle = calLinerAngleByPoint(handDirection, palmPoint);
	z_angle += PI;
	if (z_angle >= 2 * PI)
		z_angle = z_angle - 2 * PI;
	finger3ds[PALM_ANGLE].z = z_angle;

	int mini_r = 0.7*handRadius;

	if (palmPoint.x < mini_r || palmPoint.x >= 640 - mini_r || palmPoint.y < mini_r || palmPoint.y >= 480 - mini_r)
		return;

	ushort depth_x1 = rawDepthFrame.at<ushort>(palmPoint.y - mini_r, palmPoint.x);
	ushort depth_x2 = rawDepthFrame.at<ushort>(palmPoint.y + mini_r, palmPoint.x);
	
	ushort depth_y1 = rawDepthFrame.at<ushort>(palmPoint.y, palmPoint.x - mini_r);
	ushort depth_y2 = rawDepthFrame.at<ushort>(palmPoint.y, palmPoint.x + mini_r);
	
	double dz_x = depth_x2 - depth_x1;
	double dy = 2 * mini_r;
	double x_angle = atan(dz_x / dy);

	double dz_y = depth_y2 - depth_y1;
	double dx = 2 * mini_r;
	double y_angle = atan(dz_y / dx);
	
	finger3ds[PALM_ANGLE].x = x_angle;
	//finger3ds[PALM_ANGLE].y = y_angle;
	
	/*cv::circle(handLayerAbs, cv::Point(palmPoint.x, palmPoint.y - mini_r), 3, cv::Scalar(0, 0, 255), -1);
	cv::circle(handLayerAbs, cv::Point(palmPoint.x, palmPoint.y + mini_r), 3, cv::Scalar(0, 0, 255), -1);

	cv::circle(handLayerAbs, cv::Point(palmPoint.x - mini_r, palmPoint.y), 3, cv::Scalar(0, 0, 255), -1);
	cv::circle(handLayerAbs, cv::Point(palmPoint.x + mini_r, palmPoint.y), 3, cv::Scalar(0, 0, 255), -1);
*/
}

void Application::assignFingerId()
{
	FingerSorterBySideHand fingerSorter;
	fingerSorter.handDirection = handDirection;
	sort(fingerPoints.begin(), fingerPoints.end(), fingerSorter);
	int thumb_idx = -1;
	vector<int> inside_idx;
	
	for (int i = 0; i < fingerPoints.size(); i++)
	{
		cv::Point pi = fingerPoints[i];
		double di = calPointLineDistance(handDirection, pi);
		if (di >= 2.4*handRadius && thumb_idx == -1)
		{
			thumb_idx = i;
			
		}
		double dpi = calDistance(palmPoint, pi);
		if (dpi < handRadius)
		{
			inside_idx.push_back(i);
			cv::circle(handLayerAbs, pi, 6, cv::Scalar(255, 255, 0), 2);
		}
	}

	if (thumb_idx == -1 && inside_idx.size() != 0)
	{
		thumb_idx = inside_idx[0];
	}
	if (thumb_idx != -1)
	{
		cv::circle(handLayerAbs, fingerPoints[thumb_idx], 6, cv::Scalar(128, 128, 128), 2);
		finger2ds[0] = fingerPoints[thumb_idx];
	}

	int finger_idx = 1;
	for (int i = 0; i < fingerPoints.size() && finger_idx < 5; i++)
	{
		if (i != thumb_idx)
		{
			finger2ds[finger_idx] = fingerPoints[i];
			finger_idx++;
		}
	}
	finger2ds[5] = palmPoint;

	for (int i = 0; i < 6; i++)
	{
		finger3ds[i] = convertPoint2dTo3D(finger2ds[i]);
	}
}

void Application::displayResult()
{
	handMask.copyTo(handLayerAbs2);
	cv::cvtColor(handLayerAbs2, handLayerAbs2, cv::COLOR_GRAY2BGR);
	/*for (int i = 0; i < fingerPointsAbs.size(); i++)
	{
		cv::circle(handLayerAbs2, fingerPointsAbs[i], 4, cv::Scalar(0, 0, 255), -1);
	}*/
	cv::circle(handLayerAbs2, finger2ds[FINGER_THUMB], 4, THUMB_COLOR, 2);
	cv::circle(handLayerAbs2, finger2ds[FINGER_INDEX], 4, INDEX_COLOR, 2);
	cv::circle(handLayerAbs2, finger2ds[FINGER_MIDDLE], 4, MIDDLE_COLOR, 2);
	cv::circle(handLayerAbs2, finger2ds[FINGER_RING], 4, RING_COLOR, 2);
	cv::circle(handLayerAbs2, finger2ds[FINGER_LITTLE], 4, LITTLE_COLOR, 2);
	cv::circle(handLayerAbs2, palmPoint, 4, PALM_COLOR, 2);

	cv::circle(handLayerAbs2, palmPoint, handRadius, cv::Scalar(0, 255, 0), 2);
}

void Application::displayLabel()
{
	cv::circle(handLayerAbs2, cv::Point(480, 7), 4, THUMB_COLOR, -1);
	cv::circle(handLayerAbs2, cv::Point(480, 22), 4, INDEX_COLOR, -1);
	cv::circle(handLayerAbs2, cv::Point(480, 37), 4, MIDDLE_COLOR, -1);
	cv::circle(handLayerAbs2, cv::Point(480, 52), 4, RING_COLOR, -1);
	cv::circle(handLayerAbs2, cv::Point(480, 67), 4, LITTLE_COLOR, -1);
	cv::circle(handLayerAbs2, cv::Point(480, 82), 4, PALM_COLOR, -1);

	cv::putText(handLayerAbs2, "Thumb", cv::Point(490, 15), cv::FONT_HERSHEY_COMPLEX, 0.5, THUMB_COLOR, 2);
	cv::putText(handLayerAbs2, "Index", cv::Point(490, 30), cv::FONT_HERSHEY_COMPLEX, 0.5, INDEX_COLOR, 2);
	cv::putText(handLayerAbs2, "Middle", cv::Point(490, 45), cv::FONT_HERSHEY_COMPLEX, 0.5, MIDDLE_COLOR, 2);
	cv::putText(handLayerAbs2, "Ring", cv::Point(490, 60), cv::FONT_HERSHEY_COMPLEX, 0.5, RING_COLOR, 2);
	cv::putText(handLayerAbs2, "Little", cv::Point(490, 75), cv::FONT_HERSHEY_COMPLEX, 0.5, LITTLE_COLOR, 2);
	cv::putText(handLayerAbs2, "Palm", cv::Point(490, 90), cv::FONT_HERSHEY_COMPLEX, 0.5, PALM_COLOR, 2);
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
		c = -p1.x;
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

cv::Vec2d Application::calParalellLine(cv::Vec2d l, cv::Point p)
{
	double m = l[0];
	double mn = m;
	double cn = 0;
	if (m == 0) {
		cn = p.y;
	}
	else if (m == HUGE_VAL) {
		cn = -p.x;
	}
	else {
		cn = p.y - m * p.x;
	}
	return cv::Vec2d(mn, cn);
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
	double k, h, x, y;
	
	if (angle < 0.5*PI) {
		// Q3
		h = radius * sin(angle);
		k = radius * cos(angle);
		x = origin.x - h;
		y = origin.y + k;
	}
	else if (angle < PI) {
		// Q2
		k = radius * sin(angle - 0.5*PI);
		h = radius * cos(angle - 0.5*PI);
		x = origin.x - h;
		y = origin.y - k;
	}
	else if (angle < 1.5*PI) {
		// Q1
		h = radius * sin(angle - PI);
		k = radius * cos(angle - PI);
		x = origin.x + h;
		y = origin.y - k;
	}
	else {
		// Q4
		k = radius * sin(angle - 1.5*PI);
		h = radius * cos(angle - 1.5*PI);
		x = origin.x + h;
		y = origin.y + k;
	}
	
	return cv::Point(x, y);
}

double Application::calAnglePoint(cv::Point origin, cv::Point p)
{
	double theta = 0;
	double h = abs(p.x - origin.x);
	double k = abs(p.y - origin.y);

	// Q3 > Q2 > Q1 > Q4

	if (p.x <= origin.x) {
		if (p.y <= origin.y) {
			// Quadrand Pixel Space 2
			theta = (0.5*PI) + atan(k / h);
		}
		else {
			// Q3
			theta = atan(h / k);
		}
	}
	else {
		if (p.y <= origin.y) {
			// Q1
			theta = PI + atan(h / k);
		}
		else {
			// Q4
			theta = (1.5*PI) + atan(k / h);
		}
	}
	
	return theta;
}

double Application::calLinerAngleByPoint(cv::Vec2d l, cv::Point p)
{
	double m = l[0];
	if (m == 0)
		return 0.5*PI;
	if (m == HUGE_VAL)
		return PI;
	if (m < 0) {
		return atan(abs(1/m)) + PI;
	}
	if (m > 0)
		return atan(abs(m)) + 0.5*PI;
	return 0.0;
}

double Application::calPointLineDistance(cv::Vec2d l, cv::Point p)
{
	double A = l[0];
	double B = -1;
	double C = l[1];
	double d = abs(A*p.x + B * p.y + C) / sqrt(pow(A, 2) + pow(B, 2));
	return d;
}

double Application::calAngleOf3Points(cv::Point start, cv::Point farP, cv::Point end)
{
	double d_sf = calDistance(start, farP);
	double d_se = calDistance(start, end);
	double d_ef = calDistance(farP, end);

	double angle = acos((pow(d_sf, 2) + pow(d_ef, 2) - pow(d_se, 2)) / (2 * d_sf * d_ef));
	return 0.0;
}

vector<cv::Point> Application::findLargestContour(cv::Mat in)
{
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierachy;
	cv::findContours(in, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	double larea = 0;
	int lindex = -1;
	vector<cv::Point> largestContour;
	for (int i = 0; i < contours.size(); i++)
	{
		double a = cv::contourArea(contours[i]);
		if (a > larea) {
			larea = a;
			lindex = i;
		}
	}

	if (lindex != -1)
		largestContour = contours[lindex];
	return largestContour;
}

vector<cv::Point> Application::findConcavePoints(vector<cv::Point> contour, vector<int> hull, int threshold)
{
	vector<cv::Point> concavePoints;
	vector<cv::Vec4i> defect;
	cv::convexityDefects(contour, hull, defect);

	for (int i = 0; i < defect.size(); i++) {
		cv::Vec4i v = defect[i];
		int depth = v[3];
		if (depth < threshold) {
			continue;
		}

		cv::Point farPoint = contour[v[2]];
		concavePoints.push_back(farPoint);
	}
	return concavePoints;
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

cv::Point Application::EL3_findMinConcave(vector<cv::Point> concavePoints)
{
	if (concavePoints.size() == 0)
		return cv::Point(0, 0);
	cv::Point minConcave;
	int minConcaveX = 640;
	minConcave = concavePoints[0];
	for (int i = 0; i < concavePoints.size(); i++)
	{
		if (concavePoints[i].x < minConcaveX) {
			minConcaveX = concavePoints[i].x;
			minConcave = concavePoints[i];
		}
	}
	return minConcave;
}

void Application::EL3_buildTable(cv::Mat in, cv::Rect boundingBox, vector<bool>& acceptTransitionTable, vector<bool>& acceptLengthTable)
{
	int start_y = boundingBox.y;
	int end_y = boundingBox.y + boundingBox.height;
	int start_x = boundingBox.x - SCAN_PALM_PADDING;
	int end_x = boundingBox.x + boundingBox.width + SCAN_PALM_PADDING;
	for (int y = start_y; y < end_y; y++)
	{
		// scan from top-bounding to bottom-bounding
		uchar* handPalmRow = handLayer3.ptr<uchar>(y);
		int changing = 0;
		int prev = 0;
		int countNonZero = 0;

		// scan from left-padding to right-padding
		int x = start_x;
		while (x < end_x) {
			int cur = handPalmRow[x];
			if (cur == 255 || cur == 128)
				countNonZero++;

			if (cur == 128) cur = 0;

			if (cur != prev) {
				// scan hole, find the same prev in next
				int i = 1;
				bool except = false;
				while (i <= SCAN_PALM_EXCEPT_HOLE) {
					int next = handPalmRow[x + i];
					if (next == 128) next = 0;
					if (next == prev) {
						except = true;
						break;
					}
					i++;
				}

				x = x + i;
				if (!except) {
					prev = cur;
					changing++;
				}
				continue;
			}

			x++;
		}

		if (changing <= 2) {
			acceptTransitionTable[y - boundingBox.y] = true;
		}
		if (countNonZero >= 1.6*handRadius) {
			acceptLengthTable[y - boundingBox.y] = true;
		}
	}
}

cv::Vec2i Application::EL3_findMaxRegion(vector<bool> acceptTransitionTable, vector<bool> acceptLengthTable)
{
	vector<cv::Vec2i> regions;
	int start = -1;
	int end = -1;
	bool prev = false;

	// create regions
	int i = 0;
	while (i < acceptTransitionTable.size()) {
		bool at = acceptTransitionTable[i];
		bool al = acceptLengthTable[i];
		if (at && al) {
			if (start == -1) {
				start = i;
			}
		}
		else {
			if (start != -1) {
				bool except = false;
				int j = 1;
				while (j <= SCAN_TABLE_EXCEPT_HOLE && i+j < acceptTransitionTable.size()) {
					bool atj = acceptTransitionTable[i + j];
					bool alj = acceptLengthTable[i + j];
					
					if (atj && alj) {
						except = true;
						break;
					}

					j++;
				}
				
				if (!except) {
					end = i;
					regions.push_back(cv::Vec2i(start, end));
					start = -1;
					end = -1;
				}

				i = i + j;
				continue;
			}
		}

		i++;
	}
	
	if (start != -1) {
		regions.push_back(cv::Vec2i(start, acceptTransitionTable.size() - 1));
	}

	// find max region
	int maxValue = 0;
	cv::Vec2i maxRegion = cv::Vec2i(0, 0);
	for (int i = 0; i < regions.size(); i++)
	{
		cv::Vec2i region = regions[i];
		if (region[1] - region[0] > maxValue) {
			maxValue = region[1] - region[0];
			maxRegion = region;
		}
	}
	return maxRegion;
}

cv::Point Application::EL3_findRegionCenter(cv::Mat in, cv::Rect region)
{
	cv::Mat mask = cv::Mat::zeros(in.size(), CV_8UC1);
	cv::Mat sub;
	cv::rectangle(mask, region, cv::Scalar(255), -1);
	cv::bitwise_and(in, mask, sub);
	cv::Moments m = cv::moments(sub, true);
	cv::Point center = cv::Point(m.m10 / m.m00, m.m01 / m.m00);
	
	return center;
}

int Application::EL3_countWhitePoint(cv::Mat in, cv::Point point, int radius)
{
	cv::Mat mask = cv::Mat::zeros(in.size(), CV_8UC1);
	cv::Mat test;
	cv::circle(mask, point, radius, cv::Scalar(255), -1);
	cv::bitwise_and(in, mask, test);
	int white = cv::countNonZero(test);
	return white;
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
