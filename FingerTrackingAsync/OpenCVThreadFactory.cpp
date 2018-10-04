#include "pch.h"
#include "opencv2/opencv.hpp"
#include "OpenCVThreadFactory.h"

thread OpenCVThreadFactory::bitwise_and(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst, cv::InputArray mask)
{
	return thread(cv::bitwise_and, src1, src2, dst, mask);
}

thread OpenCVThreadFactory::bitwise_or(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst, cv::InputArray mask)
{
	return thread(cv::bitwise_or, src1, src2, dst, mask);
}

thread OpenCVThreadFactory::bitwise_not(cv::InputArray src1, cv::OutputArray dst, cv::InputArray mask)
{
	return thread(cv::bitwise_not, src1, dst, mask);
}

thread OpenCVThreadFactory::inRange(cv::InputArray src, cv::InputArray lowwer, cv::InputArray upper, cv::OutputArray dst)
{
	return thread(cv::inRange, src, lowwer, upper, dst);
}
