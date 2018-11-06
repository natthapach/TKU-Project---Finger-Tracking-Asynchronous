#pragma once
#include <string>
#include <stdio.h> 
#include <stdlib.h> 
#include <io.h> 
#include <string.h> 
#include <sys/types.h> 
#include <Winsock2.h>
#include "opencv2/opencv.hpp"

using namespace std;

class AdapterCaller {
public :
	int sendData(vector<cv::Point3f> points);
protected :
	const char *HOST = "127.0.0.1";
	const char *PORT = "8085";
};
