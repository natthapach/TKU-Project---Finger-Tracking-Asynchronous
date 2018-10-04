#pragma once
#include "KinectReader.h"

class Application {
public :
	int initialize();
	void start();
private:
	KinectReader kinectReader;
};