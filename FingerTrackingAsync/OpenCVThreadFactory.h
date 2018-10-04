#pragma once
#include <thread> 
#include "opencv2/opencv.hpp"

using namespace std;



class OpenCVThreadFactory {
public :
	static thread bitwise_and(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst, cv::InputArray mask = cv::noArray());
	static thread bitwise_or(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst, cv::InputArray mask = cv::noArray());
	static thread bitwise_not(cv::InputArray src1, cv::OutputArray dst, cv::InputArray mask = cv::noArray());

	static thread inRange(cv::InputArray src, cv::InputArray lowwer, cv::InputArray upper, cv::OutputArray dst);
};
