/******************************************************************************************************
* (c) Virginia Polytechnic Insitute and State University, 2011.
******************************************************************************************************/
#include "timeRec.h"
#include <stdio.h>
#include <stdlib.h>

struct timeval timeStart;
struct timeval timeEnd;

void timerStart()
{
	gettimeofday(&timeStart, NULL);
}

void timerEnd()
{
	gettimeofday(&timeEnd, NULL);
}

//return value is ms
double elapsedTime()
{
	double deltaTime;
	deltaTime = (timeEnd.tv_sec - timeStart.tv_sec) * 1000.0 + 
				(timeEnd.tv_usec - timeStart.tv_usec) / 1000.0;
	
	return deltaTime;
}
