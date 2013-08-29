#pragma once

#include <string>

#include "Timer.h"
#include "BufferCollection.h"

template<typename T>
void WriteValue(BufferCollection& bc, const BufferCollectionKey_t& key, T value)
{
	VectorBufferTemplate<T>& buffer = bc.GetOrAddBuffer< VectorBufferTemplate<T> >(key);
	buffer.Resize(1);
	buffer.Set(0, value);
}

template<typename T>
void IncrementValue(BufferCollection& bc, const BufferCollectionKey_t& key, T value)
{
	VectorBufferTemplate<T>& buffer = bc.GetOrAddBuffer< VectorBufferTemplate<T> >(key);
	buffer.Resize(1);
	buffer.Incr(0, value);
}

template<typename T>
void WriteValue(BufferCollection& bc, const BufferCollectionKey_t& key, int index, T value)
{
	VectorBufferTemplate<T>& buffer = bc.GetOrAddBuffer< VectorBufferTemplate<T> >(key);
	buffer.Extend(index+1);
	buffer.Set(index, value);
}

template<typename T>
void IncrementValue(BufferCollection& bc, const BufferCollectionKey_t& key, int index, T value)
{
	VectorBufferTemplate<T>& buffer = bc.GetOrAddBuffer< VectorBufferTemplate<T> >(key);
	buffer.Extend(index+1);
	buffer.Incr(index, value);
}

class TimeLogger
{
public:
    TimeLogger(BufferCollection& bc, const std::string& name);
    TimeLogger(BufferCollection& bc, int nodeIndex);
    ~TimeLogger();
private:
	BufferCollection& mBufferCollection;
	const std::string mTimeName;
	const std::string mCounterName;
	const bool mRecordCounter; 
	const int mNodeIndex;
	Timer mTimer;
};