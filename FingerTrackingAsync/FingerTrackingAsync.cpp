#include "pch.h"
#include <iostream>
#include "Application.h"

Application application;

int main()
{
    std::cout << "Hello World!\n"; 
	int status = 0;
	status = application.initialize();
	if (status != 0)
		return 1;

	application.start();

	return 0;
}
