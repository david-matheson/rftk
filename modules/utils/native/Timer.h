#pragma once

#include <ctime>

class Timer
{
public:
    Timer();
    void   Restart();
    clock_t ElapsedClock() const;
    double ElapsedMilliSeconds() const;

private:
	 clock_t mStartTime;
};
