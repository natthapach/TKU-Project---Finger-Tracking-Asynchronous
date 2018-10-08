#include "pch.h"
#include "Application.h"
#include "KinectReader.h"
#include "opencv2/opencv.hpp"
#include "OpenCVThreadFactory.h"

using namespace std;

const string WINDOW_RGB = "RGB";
const string WINDOW_DEPTH = "Depth";

int Application::initialize()
{
	int status = 0;
	status = kinectReader.initialize();
	if (status != 0)
		return 1;

	cv::namedWindow(WINDOW_RGB, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(WINDOW_DEPTH, cv::WINDOW_AUTOSIZE);
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
			thread buildSkinMaskT = thread(&Application::buildSkinMask, this);
			thread buildEdgeColorT = thread(&Application::buildEdgeColor, this);
			
			buildSkinMaskT.join(); // 18ms
			int a = 0;
			buildDepthHandMaskT.join();
			int b = 0;

			//thread combineSkinDepthT = thread(&Application::combineSkinHandMask, this);
			//combineSkinDepthT.join(); // 7ms

			thread buildHand3LayersT = thread(&Application::buildHand3Layers, this);
			buildHand3LayersT.join(); //10

			evaluateHandLayer1();

			buildEdgeColorT.join();

			cv::imshow("Mask L1", handLayer1); // 14
			cv::imshow("Mask L2", handLayer2);
			cv::imshow("Mask L3", handLayer3);
			cv::imshow("Edge", edgeColorFrame);
		}

		if (tickCount == 0) {
			tickCount = cv::getTickCount();
		}
		else {
			int64 t = cv::getTickCount();
			double fpsT = cv::getTickFrequency() / (t - tickCount);
			tickCount = t;
			cout << "FPS T " << fpsT << endl;
		}

		//cv::normalize(rawDepthFrame, rawDepthFrame, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imshow(WINDOW_RGB, colorFrame);
		cv::imshow(WINDOW_DEPTH, depthFrame);

		int key = cv::waitKey(1);
		if (key == 27)
			break;
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
	cv::Canny(gray, edgeColorFrame, 50, 200, 3);
	cv::dilate(edgeColorFrame, edgeColorFrame, cv::Mat());
	cv::bitwise_not(edgeColorFrame, edgeColorFrame);
	cv::floodFill(edgeColorFrame, cv::Point(handPosX, handPosY), cv::Scalar(128));
	cv::circle(edgeColorFrame, cv::Point(handPosX, handPosY), 4, cv::Scalar(0, 0, 0), -1);
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
	ushort minDepth = 65535;
	ushort maxDepth = 0;
	for (int i = 0; i < rawDepthFrame.rows; i++) {
		ushort* rawRow = rawDepthFrame.ptr<ushort>(i);
		uchar* maskRow = handMask.ptr<uchar>(i);
		for (int j = 0; j < rawDepthFrame.cols; j++) {
			if (maskRow[j] != 0) {
				if (rawRow[j] > maxDepth)
					maxDepth = rawRow[j];
				if (rawRow[j] < minDepth && rawRow[j] != 0)
					minDepth = rawRow[j];
			}
		}
	}
	int range = maxDepth - minDepth;
	int l1_max = minDepth + (range * 0.2);
	int l2_max = minDepth + (range * 0.6);
	cv::threshold(rawDepthFrame, handLayer1, l1_max, 65535, cv::THRESH_BINARY_INV);
	handLayer1.convertTo(handLayer1, CV_8UC1, 255.0 / 65535);
	cv::bitwise_and(handMask, handLayer1, handLayer1);

	cv::threshold(rawDepthFrame, handLayer2, l2_max, 65535, cv::THRESH_BINARY_INV);
	handLayer2.convertTo(handLayer2, CV_8UC1, 255.0 / 65535);
	cv::bitwise_and(handMask, handLayer2, handLayer2);

	handMask.copyTo(handLayer3);

	int a = 0;
}

void Application::evaluateHandLayer1()
{
	/*handLayer1Corners.clear();
	cv::Mat corner;
	cv::cornerHarris(handLayer1, corner, 8, 5, 0.04, cv::BORDER_DEFAULT);
	cv::normalize(corner, corner, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	for (int j = 0; j < corner.rows; j++) {
		float* cornerRow = corner.ptr<float>(j);
		for (int i = 0; i < corner.cols; i++) {
			if (cornerRow[i] > 200) {
				if (handLayer1.ptr<uchar>(j)[i] > 0)
					handLayer1Corners.push_back(cv::Point(i, j));
			}
		}
	}*/

	vector<vector<cv::Point>> contours;
	
	vector<cv::Vec4i> hierachy, hierachyI;
	cv::findContours(handLayer1, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	vector<vector<cv::Point>> convexHull(contours.size());
	vector<vector<int>> convexHullI(contours.size());
	vector<vector<double>> hullD1(contours.size());
	vector<vector<double>> contourD1(contours.size());
	vector<vector<cv::Vec4i>> defects(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		cv::convexHull(contours[i], convexHull[i]);
		cv::convexHull(contours[i], convexHullI[i]);
		cv::convexityDefects(contours[i], convexHullI[i], defects[i]);
	}

	cv::cvtColor(handLayer1, handLayer1, cv::COLOR_GRAY2BGR);
	/*for (int i = 0; i < contours.size(); i++) {
		for (int j = 0; j < contours[i].size(); j++) {
			cv::circle(handLayer1, contours[i][j], 1, cv::Scalar(0, 0, 255), -1);
		}
	}*/
	//for (int i = 0; i < handLayer1Corners.size(); i++) {
	//	cv::circle(handLayer1, handLayer1Corners[i], 2, cv::Scalar(0, 0, 255), -1);
	//}
	for (int i = 0; i < defects.size(); i++) {
		for (int j = 0; j < defects[i].size(); j++) {
			if (defects[i][j][3] > 10) {
				cv::Point startPoint = contours[i][defects[i][j][0]];
				cv::Point endPoint = contours[i][defects[i][j][1]];
				cv::Point farPoint = contours[i][defects[i][j][2]];
				
				/*cv::line(handLayer1, farPoint, startPoint, cv::Scalar(0, 0, 255), 1);
				cv::line(handLayer1, farPoint, endPoint, cv::Scalar(0, 0, 255), 1);
				cv::circle(handLayer1, farPoint, 2, cv::Scalar(0, 0, 255), -1);*/
			}
		}
	}
	
	for (int i = 0; i < convexHull.size(); i++) {
		//cv::drawContours(handLayer1, convexHull, i, cv::Scalar(0, 0, 255), 1, 8, cv::Vec4i(), 0, cv::Point());
		hullD1[i] = vector<double>(convexHull[i].size());
		for (int j = 0; j < convexHull[i].size(); j++) {
			cv::Point p0 = convexHull[i][j];
			cv::Point p1 = convexHull[i][(j - 1) % convexHull[i].size()];
			cv::Point p2 = convexHull[i][(j + 1) % convexHull[i].size()];
			double m1 = ((double)(p0.y - p1.y)) / (p0.x - p1.x);
			double m2 = ((double)(p2.y - p0.y)) / (p2.x - p0.x);
			hullD1[i][j] = m2 / m1;
			cv::line(handLayer1, p0, p1, cv::Scalar(0, 0, 255), 1);
			/*if (hullD1[i][j] == 0) {
				cv::circle(handLayer1, convexHull[i][j], 2, cv::Scalar(255, 0, 0), -1);
			}*/
			/*if (hullD1[i][j] < 0) {
				cv::circle(handLayer1, convexHull[i][j], 2, cv::Scalar(255, 0, 0), -1);
				
			} else if (hullD1[i][j] > 0) {
				cv::circle(handLayer1, convexHull[i][j], 2, cv::Scalar(0, 255, 0), -1);
			}
			else {
				cv::circle(handLayer1, convexHull[i][j], 2, cv::Scalar(0, 255, 255), -1);
			}*/
			//cv::line(handLayer1, convexHull[i][j], convexHull[i][(j + 1) % convexHull[i].size()], cv::Scalar(255, 0, 0), 1);
			//cv::circle(handLayer1, convexHull[i][j], 1, cv::Scalar(255, 0, 0), -1);
		}
	}
	vector<vector<int>> contourDD(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		contourD1[i] = vector<double>(contours[i].size());
		for (int j = 0; j < contours[i].size(); j++) {
			cv::Point p1 = contours[i][(j - 1) % contours[i].size()];
			cv::Point p0 = contours[i][j];
			cv::Point p2 = contours[i][(j + 1) % contours[i].size()];
			
			double dx1 = p0.x - p1.x;
			double dy1 = p0.y - p1.y;
			double dx2 = p2.x - p0.x;
			double dy2 = p2.y - p0.y;

			double m1 = dy1 / dx1;
			double m2 = dy2 / dy2;
			
			double d1 = m2 / m1;
			contourD1[i][j] = d1;

			double zero = 0;

			if (d1 == 0) {
				//cv::circle(handLayer1, p0, 2, cv::Scalar(255, 0, 0), -1);
			}
			else if (d1 == 1.0 / zero) {
				cv::circle(handLayer1, p0, 2, cv::Scalar(0, 255, 0), -1);
				contourDD[i].push_back(j);
			}
			else if (d1 == -1.0 / zero) {
				cv::circle(handLayer1, p0, 2, cv::Scalar(0, 255, 255), -1);
				contourDD[i].push_back(-j);
			}
		}
	}

	for (int i = 0; i < contourDD.size(); i++) {
		for (int j = 0; j < contourDD[i].size(); j++) {
			int d0 = contourDD[i][j];
			int d1 = contourDD[i][(j - 1) % contourDD[i].size()];
			int d2 = contourDD[i][(j + 1) % contourDD[i].size()];
			if (d0 > 0 && d1 < 0) {
				cv::Point p0 = contours[i][d0];
				cv::Point p1 = contours[i][-d1];
				int cx = (p0.x + p1.x) / 2;
				int cy = (p0.y + p1.y) / 2;
				cv::circle(handLayer1, cv::Point(cx, cy), 4, cv::Scalar(255, 0, 0), -1);
			}
			else if (d0 > 0 && d2 < 0) {
				cv::Point p0 = contours[i][d0];
				cv::Point p2 = contours[i][-d2];
				int cx = (p0.x + p2.x) / 2;
				int cy = (p0.y + p2.y) / 2;
				cv::circle(handLayer1, cv::Point(cx, cy), 4, cv::Scalar(255, 0, 0), -1);
			}
		}
	}

	for (int i = 0; i < hullD1.size(); i++) {
		for (int j = 0; j < hullD1[i].size(); j++) {
			if (hullD1[i][j] != 0)
				continue;
			double m1 = hullD1[i][(j - 1) % hullD1[i].size()];
			double m2 = hullD1[i][(j + 1) % hullD1[i].size()];
			
			double dm = m2 / m1;

			if (dm == 0) {
				//cv::circle(handLayer1, convexHull[i][j], 4, cv::Scalar(255, 0, 0), -1);
			}
			else if (dm < 0) {
				//cv::circle(handLayer1, convexHull[i][j], 2, cv::Scalar(0, 255, 0), -1);
			}
			/*if (dm > 0) {
				cv::circle(handLayer1, convexHull[i][j], 2, cv::Scalar(0, 255, 255), -1);
			} else if(dm < 0) {
				cv::circle(handLayer1, convexHull[i][j], 2, cv::Scalar(0, 255, 0), -1);
			}*/
		}
	}
}

void Application::calculateContourArea(vector<cv::Point> contour, double * area)
{
	(*area) = cv::contourArea(contour);
}
