#include "pch.h"
#include <iostream>
#include "Application.h"
#include "AdapterCaller.h"
#include "opencv2/opencv.hpp"

Application application;
AdapterCaller adapterCaller;

int main()
{
    std::cout << "Hello World!\n"; 
	int status = 0;
	vector<cv::Point3f> points(6, cv::Point3f(0, 0, 0));
	application.setAdapterCaller(adapterCaller);
	status = application.initialize();
	//adapterCaller.sendData(points);
	
	if (status != 0)
		return 1;

	application.start();

	return 0;
}
