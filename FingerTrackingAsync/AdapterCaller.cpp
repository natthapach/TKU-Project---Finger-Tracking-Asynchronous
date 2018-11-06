#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "pch.h"
#include "AdapterCaller.h"
#include <string>
#include <stdio.h> 
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include "Application.h"

#pragma comment(lib, "Ws2_32.lib")

int AdapterCaller::sendData(vector<cv::Point3f> points)
{
	WSADATA wsaData;
	int iResult;

	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		printf("WSAStartup failed: %d\n", iResult);
		return 1;
	}

	struct addrinfo *result = NULL,
		*ptr = NULL,
		hints;


	ZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	// Resolve the server address and port
	iResult = getaddrinfo(HOST, PORT, &hints, &result);
	if (iResult != 0) {
		printf("getaddrinfo failed: %d\n", iResult);
		WSACleanup();
		return 1;
	}

	SOCKET ConnectSocket = INVALID_SOCKET;

	// Attempt to connect to the first address returned by
	// the call to getaddrinfo
	ptr = result;

	// Create a SOCKET for connecting to server
	ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);

	if (ConnectSocket == INVALID_SOCKET) {
		printf("Error at socket(): %ld\n", WSAGetLastError());
		freeaddrinfo(result);
		WSACleanup();
		return 1;
	}

	// Connect to server.
	iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
	if (iResult == SOCKET_ERROR) {
		closesocket(ConnectSocket);
		ConnectSocket = INVALID_SOCKET;
	}

	// Should really try the next address returned by getaddrinfo
	// if the connect call failed
	// But for this simple example we just free the resources
	// returned by getaddrinfo and print an error message

	freeaddrinfo(result);

	if (ConnectSocket == INVALID_SOCKET) {
		printf("Unable to connect to server!\n");
		WSACleanup();
		return 1;
	}

	int recvbuflen = 512;

	const char* bufferPtr[6];
	for (int i = 0; i < 6; i++)
	{
		char buffer_small[50];
		cv::Point3f point = points[i];
		sprintf_s(buffer_small, "%d:(%.2f, %.2f, %.2f)", i, point.x, point.y, point.z);
		bufferPtr[i] = buffer_small;
	}
	
	const char *format = "{%d:(%.2f, %.2f, %.2f)},"
		"%d:(%.2f, %.2f, %.2f),"
		"%d:(%.2f, %.2f, %.2f),"
		"%d:(%.2f, %.2f, %.2f),"
		"%d:(%.2f, %.2f, %.2f),"
		"%d:(%.2f, %.2f, %.2f)}";
	char buffer[300];
	sprintf_s(buffer, format,
		0, points[0].x, points[0].y, points[0].z,
		1, points[1].x, points[1].y, points[1].z,
		2, points[2].x, points[2].y, points[2].z,
		3, points[3].x, points[3].y, points[3].z,
		4, points[4].x, points[4].y, points[4].z,
		5, points[5].x, points[5].y, points[5].z);
	cout << buffer << endl;
	//const char *sendbuf = "this is a test";
	const char *sendbuf = buffer;
	char recvbuf[512];


	// Send an initial buffer
	iResult = send(ConnectSocket, sendbuf, (int)strlen(sendbuf), 0);
	if (iResult == SOCKET_ERROR) {
		printf("send failed: %d\n", WSAGetLastError());
		closesocket(ConnectSocket);
		WSACleanup();
		return 1;
	}

	printf("Bytes Sent: %ld\n", iResult);

	// shutdown the connection for sending since no more data will be sent
	// the client can still use the ConnectSocket for receiving data
	iResult = shutdown(ConnectSocket, SD_SEND);
	if (iResult == SOCKET_ERROR) {
		printf("shutdown failed: %d\n", WSAGetLastError());
		closesocket(ConnectSocket);
		WSACleanup();
		return 1;
	}

	// Receive data until the server closes the connection
	do {
		iResult = recv(ConnectSocket, recvbuf, recvbuflen, 0);
		if (iResult > 0)
			printf("Bytes received: %d\n", iResult);
		else if (iResult == 0)
			printf("Connection closed\n");
		else
			printf("recv failed: %d\n", WSAGetLastError());
	} while (iResult > 0);

	// shutdown the send half of the connection since no more data will be sent
	iResult = shutdown(ConnectSocket, SD_SEND);
	if (iResult == SOCKET_ERROR) {
		printf("shutdown failed: %d\n", WSAGetLastError());
		closesocket(ConnectSocket);
		WSACleanup();
		return 1;
	}


	// cleanup
	closesocket(ConnectSocket);
	WSACleanup();
}
