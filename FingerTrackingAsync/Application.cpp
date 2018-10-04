#include "pch.h"
#include "Application.h"
#include "KinectReader.h"
#include "opencv2/opencv.hpp"

int Application::initialize()
{
	return kinectReader.initialize();
}

void Application::start()
{
	while (true) {
		kinectReader.readRGBFrame();
		kinectReader.readDepthFrame();
		cv::Mat colorFrame = kinectReader.getRGBFrame();
		cv::Mat depthFrame = kinectReader.getDepthFrame();
		cv::namedWindow("RGB", cv::WINDOW_AUTOSIZE);
		cv::imshow("RGB", colorFrame);
		cv::imshow("Depth", depthFrame);
		int key = cv::waitKey(5);
		if (key == 27)
			break;
	}
	
	cv::destroyAllWindows();
}
