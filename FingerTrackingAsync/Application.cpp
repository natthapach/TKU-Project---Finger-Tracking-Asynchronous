#include "pch.h"
#include "Application.h"
#include "KinectReader.h"
#include "opencv2/opencv.hpp"

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
		cv::Mat colorFrame = kinectReader.getRGBFrame();
		cv::Mat depthFrame = kinectReader.getDepthFrame();
		
		if (kinectReader.isHandTracking()) {

		}

		cv::imshow(WINDOW_RGB, colorFrame);
		cv::imshow(WINDOW_DEPTH, depthFrame);

		int key = cv::waitKey(5);
		if (key == 27)
			break;
	}
	
	cv::destroyAllWindows();
}
