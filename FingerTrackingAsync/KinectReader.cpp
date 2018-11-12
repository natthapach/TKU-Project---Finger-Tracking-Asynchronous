#include "pch.h"
#include "KinectReader.h"

using namespace std;

int KinectReader::initialize()
{
	
	openni::Status statusNi = openni::STATUS_OK;
	nite::Status statusNite = nite::STATUS_OK;
	
	// open Nite HandTracker
	statusNite = nite::NiTE::initialize();
	if (statusNite != nite::STATUS_OK)
		return 1;

	statusNite = handTracker.create();
	if (statusNite != nite::STATUS_OK)
		return 1;

	statusNite = handTracker.startGestureDetection(nite::GESTURE_HAND_RAISE);
	if (statusNite != nite::STATUS_OK)
		return 1;

	// initial openni
	statusNi = openni::OpenNI::initialize();
	if (statusNi != openni::STATUS_OK)
		return 1;

	statusNi = device.open(openni::ANY_DEVICE);
	if (statusNi != openni::STATUS_OK)
		return 1;
	
	

	// open depth stream
	statusNi = depthStream.create(device, openni::SENSOR_DEPTH);
	if (statusNi != openni::STATUS_OK)
		return 1;

	// open rgb stream
	statusNi = colorStream.create(device, openni::SENSOR_COLOR);
	if (statusNi != openni::STATUS_OK)
		return 1;

	openni::VideoMode vmod;
	vmod.setFps(30);
	vmod.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
	vmod.setResolution(640, 480);
	statusNi = colorStream.setVideoMode(vmod);
	if (statusNi != openni::STATUS_OK)
		return 1;

	statusNi = colorStream.start();
	if (statusNi != openni::STATUS_OK)
		return 1;

	/*statusNi = device.setDepthColorSyncEnabled(true);
	statusNi = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
*/
	return 0;
}

void KinectReader::readDepthFrame()
{
	readDepthThread = thread(&KinectReader::asyncReadDepthFrame, this);
}

void KinectReader::readRGBFrame()
{
	readRGBThread = thread(&KinectReader::asyncReadRGBFrame, this);
}

cv::Mat KinectReader::getDepthFrame()
{
	if (readDepthThread.joinable())
		readDepthThread.join();
	depthFrame = cv::Mat(480, 640, CV_8UC3, &img);
	//cv::flip(depthFrame, depthFrame, 1);
	
	return depthFrame;
}

cv::Mat KinectReader::getRGBFrame()
{
	if (readRGBThread.joinable())
		readRGBThread.join();
	bool t = readRGBThread.joinable();
	return colorFrame;
}

cv::Mat KinectReader::getDepthHandMask()
{
	depthHandMask = cv::Mat(480, 640, CV_8UC1, &mask);
	//cv::flip(depthHandMask, depthHandMask, 1);
	return depthHandMask;
}

cv::Mat KinectReader::getRawDepthFrame()
{
	cv::Mat rawDepthFrame = cv::Mat(480, 640, CV_16UC1, &depthRaw);
	//cv::flip(rawDepthFrame, rawDepthFrame, 1);
	return rawDepthFrame;
}

bool KinectReader::isHandTracking()
{
	return numberOfHands > 0 && handDepth > 0;
}

float KinectReader::getHandPosX()
{
	return handPosX;
}

float KinectReader::getHandPosY()
{
	return handPosY;
}

int KinectReader::getHandDepth()
{
	return handDepth;
}

int KinectReader::getDepthHandRange()
{
	return RANGE;
}

cv::Point KinectReader::getHandPoint()
{
	return cv::Point(handPosX, handPosY);
}

int KinectReader::getHandRadius(int mm)
{
	if (!isHandTracking())
		return 0;
	float handWorldX, handWorldY, handWorldZ;
	openni::CoordinateConverter::convertDepthToWorld(depthStream, handPosX, handPosY, handDepth, &handWorldX, &handWorldY, &handWorldZ);
	handWorldY -= mm;
	float radiusX, radiusY, radiusZ;
	openni::CoordinateConverter::convertWorldToDepth(depthStream, handWorldX, handWorldY, handWorldZ, &radiusX, &radiusY, &radiusZ);
	int radius = (int) abs(radiusY - handPosY);
	if (radius < 0)
		return 0;
	return radius;
}

vector<cv::Point> KinectReader::getHandPoints()
{
	return handPositions;
}

void KinectReader::convertDepthToColor(int x, int y, int z, int * cx, int * cy)
{
	openni::CoordinateConverter::convertDepthToColor(depthStream, colorStream, x, y, z, cx, cy);
}

void KinectReader::convertDepthToWorld(float x, float y, float z, float * wx, float * wy, float * wz)
{
	openni::CoordinateConverter::convertDepthToWorld(depthStream, x, y, z, wx, wy, wz);
}

void KinectReader::asyncReadRGBFrame()
{
	uchar img[480][640][3];
	openni::Status status = openni::STATUS_OK;
	openni::VideoStream* streamPointer = &colorStream;
	int streamReadyIndex;
	status = openni::OpenNI::waitForAnyStream(&streamPointer, 1, &streamReadyIndex, 500);

	if (status != openni::STATUS_OK && streamReadyIndex != 0)
		return;

	openni::VideoFrameRef newFrame;
	status = colorStream.readFrame(&newFrame);
	if (status != openni::STATUS_OK && !newFrame.isValid())
		return;

	for (unsigned int y = 0; y < 480; y++) {
		for (unsigned int x = 0; x < 640; x++) {
			OniRGB888Pixel* streamPixel = (OniRGB888Pixel*)((char*)newFrame.getData() + (y * newFrame.getStrideInBytes())) + x;
			img[y][x][0] = streamPixel->b;
			img[y][x][1] = streamPixel->g;
			img[y][x][2] = streamPixel->r;
		}
	}

	cv::Mat(480, 640, CV_8UC3, &img).copyTo(colorFrame);
	int a = 2;
}

void KinectReader::asyncReadDepthFrame()
{
	if (!handTracker.isValid())
		return;

	nite::Status status = nite::STATUS_OK;

	try {
		status = handTracker.readFrame(&handsFrame);
	}
	catch(int e) {
		cout << "Unhandle Exception from Hand Tracker Read Frame : " << e << endl;
		return;
	}
	if (status != nite::STATUS_OK || !handsFrame.isValid())
		return;


	const nite::Array<nite::GestureData>& gestures = handsFrame.getGestures();
	for (int i = 0; i < gestures.getSize(); ++i) {
		if (gestures[i].isComplete()) {
			nite::HandId handId;
			handTracker.startHandTracking(gestures[i].getCurrentPosition(), &handId);
		}
	}

	openni::VideoFrameRef depthFrame = handsFrame.getDepthFrame();

	int numberOfPoints = 0;
	int numberOfHandPoints = 0;
	calDepthHistogram(depthFrame, &numberOfPoints, &numberOfHandPoints);
	modifyImage(depthFrame, numberOfPoints, numberOfHandPoints);
	settingHandValue();

	//imageFrame = cv::Mat(480, 640, CV_8UC3, &img);
	//maskFrame = cv::Mat(480, 640, CV_8UC1, &mask);
	
}

void KinectReader::calDepthHistogram(openni::VideoFrameRef depthFrame, int * numberOfPoints, int * numberOfHandPoints)
{
	*numberOfPoints = 0;
	*numberOfHandPoints = 0;

	memset(depthHistogram, 0, sizeof(depthHistogram));
	for (int y = 0; y < depthFrame.getHeight(); ++y)
	{
		openni::DepthPixel* depthCell = (openni::DepthPixel*)
			(
			(char*)depthFrame.getData() +
				(y * depthFrame.getStrideInBytes())
				);
		for (int x = 0; x < depthFrame.getWidth(); ++x, ++depthCell)
		{
			if (*depthCell != 0)
			{
				depthHistogram[*depthCell]++;
				(*numberOfPoints)++;

				if (handDepth > 0 && numberOfHands > 0) {
					if (handDepth - RANGE <= *depthCell && *depthCell <= handDepth + RANGE)
						(*numberOfHandPoints)++;
				}
			}
		}
	}
	for (int nIndex = 1; nIndex < sizeof(depthHistogram) / sizeof(int); nIndex++)
	{
		depthHistogram[nIndex] += depthHistogram[nIndex - 1];
	}
}

void KinectReader::modifyImage(openni::VideoFrameRef depthFrame, int numberOfPoints, int numberOfHandPoints)
{
	for (unsigned int y = 0; y < 480; y++) {
		for (unsigned int x = 0; x < 640; x++) {
			openni::DepthPixel* depthPixel = (openni::DepthPixel*)
				((char*)depthFrame.getData() + (y*depthFrame.getStrideInBytes())) + x;
			depthRaw[y][x] = *depthPixel;

			if (*depthPixel != 0) {
				uchar depthValue = (uchar)(((float)depthHistogram[*depthPixel] / numberOfPoints) * 255);
				img[y][x][0] = 255 - depthValue;
				img[y][x][1] = 255 - depthValue;
				img[y][x][2] = 255 - depthValue;
			}
			else {
				img[y][x][0] = 0;
				img[y][x][1] = 0;
				img[y][x][2] = 0;
			}

			if (handDepth != 0 && numberOfHands > 0) {
				if (*depthPixel != 0 && (handDepth - RANGE <= *depthPixel && *depthPixel <= handDepth + RANGE)) {
					mask[y][x] = 128;
				}
				else {
					mask[y][x] = 0;
				}
			}
			else {
				mask[y][x] = 0;
			}
		}
	}
}

void KinectReader::settingHandValue()
{
	handPositions.clear();
	const nite::Array<nite::HandData>& hands = handsFrame.getHands();

	for (int i = 0; i < hands.getSize(); i++) {
		nite::HandData hand = hands[i];

		if (hand.isTracking()) {
			nite::Point3f position = hand.getPosition();
			float x, y;
			handTracker.convertHandCoordinatesToDepth(
				hand.getPosition().x,
				hand.getPosition().y,
				hand.getPosition().z,
				&x, &y
			);
			handPosX = x;
			handPosY = y;
			
			handPositions.push_back(cv::Point(x, y));
			openni::VideoFrameRef depthFrame = handsFrame.getDepthFrame();
			openni::DepthPixel* depthPixel = (openni::DepthPixel*) ((char*)depthFrame.getData() + ((int)y * depthFrame.getStrideInBytes())) + (int)x;
			handDepth = *depthPixel;
		}

		if (hand.isLost())
			numberOfHands--;
		if (hand.isNew())
			numberOfHands++;
	}
}
