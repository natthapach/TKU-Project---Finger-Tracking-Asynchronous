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
	void testSend(int n);
	int sendData(map<int, cv::Point3f> points);
protected :
	const char *HOST = "127.0.0.1";
	const char *PORT = "8085";
};
