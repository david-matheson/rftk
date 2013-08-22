#include "Timer.h"

Timer::Timer()
{
	mStartTime = clock(); 
}

void Timer::Restart() 
{ 
	mStartTime = clock(); 
}

clock_t Timer::ElapsedClock() const
{
	return clock() - mStartTime;
}

double Timer::ElapsedMilliSeconds() const
{
	const double milliSecondsPerClock = 1000.0 / CLOCKS_PER_SEC;
    return double(clock() - mStartTime) * milliSecondsPerClock;
}