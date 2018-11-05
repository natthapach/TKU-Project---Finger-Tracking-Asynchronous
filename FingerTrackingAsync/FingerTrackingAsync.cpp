#include "pch.h"
#include <iostream>
#include "Application.h"
#include "AdapterCaller.h"

Application application;
AdapterCaller adapterCaller;

int main()
{
    std::cout << "Hello World!\n"; 
	int status = 0;
	status = application.initialize();
	//adapterCaller.testSend(5);
	if (status != 0)
		return 1;

	application.start();

	return 0;
}
