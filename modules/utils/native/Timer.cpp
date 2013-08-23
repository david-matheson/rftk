#include "Timer.h"

Timer::Timer()
: mStartTime(boost::chrono::system_clock::now())
{
}

void Timer::Restart() 
{ 
	mStartTime = boost::chrono::system_clock::now(); 
}


double Timer::ElapsedMilliSeconds() const
{
	boost::chrono::duration<double> elapsed = boost::chrono::system_clock::now() - mStartTime;
	return elapsed.count();
}