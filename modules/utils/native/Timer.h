#pragma once

#include <boost/chrono.hpp>

class Timer
{
public:
    Timer();
    void   Restart();
    double ElapsedMilliSeconds() const;

private:
	 boost::chrono::system_clock::time_point mStartTime;
};
