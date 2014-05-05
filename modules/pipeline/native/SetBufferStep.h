#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// SetBufferStep writes a buffer to OutputBufferId so it can be used by later
// steps.  It can either written everytime the pipeline step is run or
// only when OutputBufferId has not already been set.
//
// ----------------------------------------------------------------------------
template <class DataBufferType>
class SetBufferStep: public PipelineStepI
{
public:
    SetBufferStep(const DataBufferType& buffer, SetRule setRule );
    virtual ~SetBufferStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffer
    const BufferId OutputBufferId;
private:
    DataBufferType mBuffer;
    SetRule mSetRule;
};


template <class DataBufferType>
SetBufferStep<DataBufferType>::SetBufferStep(const DataBufferType& buffer, SetRule setRule)
: PipelineStepI("SetBufferStep")
, OutputBufferId(GetBufferId("SetBufferStep"))
, mBuffer(buffer)
, mSetRule(setRule)
{}

template <class DataBufferType>
SetBufferStep<DataBufferType>::~SetBufferStep()
{}

template <class DataBufferType>
PipelineStepI* SetBufferStep<DataBufferType>::Clone() const
{
    SetBufferStep* clone = new SetBufferStep<DataBufferType>(*this);
    return clone;
}

template <class DataBufferType>
void SetBufferStep<DataBufferType>::ProcessStep(const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection,
                                        boost::mt19937& gen,
                                        BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(readCollection);
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    if(!writeCollection.HasBuffer<DataBufferType>(OutputBufferId) || mSetRule == EVERY_PROCESS)
    {
        writeCollection.AddBuffer(OutputBufferId, mBuffer);
    }
}
