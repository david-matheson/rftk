#include "BufferCollectionUtils.h"

TimeLogger::TimeLogger(BufferCollection& bc, const std::string& name)
: mBufferCollection(bc)
, mTimeName("Time-"+name)
, mCounterName(mTimeName+"-Count")
, mRecordCounter(true)
, mNodeIndex(0)
, mTimer()
{}

TimeLogger::TimeLogger(BufferCollection& bc, int nodeIndex)
: mBufferCollection(bc)
, mTimeName("Time-PerNode")
, mRecordCounter(false)
, mNodeIndex(nodeIndex)
, mTimer()
{}

TimeLogger::~TimeLogger()
{
	const double delta = mTimer.ElapsedMilliSeconds();
	IncrementValue<double>(mBufferCollection, mTimeName, mNodeIndex, delta);
	if(mRecordCounter)
	{
		IncrementValue<int>(mBufferCollection, mCounterName, mNodeIndex, 1);
	}
}
