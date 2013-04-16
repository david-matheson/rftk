#pragma once

#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// SetBufferStep writes a buffer to OutputBufferId so it can be used by later
// steps.  It can either written everytime the pipeline step is run or
// only when OutputBufferId has not already been set.
//
// ----------------------------------------------------------------------------
enum SetRule
{
    WHEN_NEW,
    EVERY_PROCESS
};

template <class BufType>
class SetBufferStep: public PipelineStepI
{
public:
    SetBufferStep(const BufType& buffer, SetRule setRule );
    virtual ~SetBufferStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const VectorBufferTemplate<long long> indices,
                                const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

    // Read only output buffer
    const UniqueBufferId::BufferId OutputBufferId;
private:
    BufType mBuffer;
    SetRule mSetRule;
};


template <class BufType>
SetBufferStep<BufType>::SetBufferStep(const BufType& buffer, SetRule setRule)
: OutputBufferId(UniqueBufferId::GetBufferId("SetBufferStep"))
, mBuffer(buffer)
, mSetRule(setRule)
{}

template <class BufType>
SetBufferStep<BufType>::~SetBufferStep()
{}

template <class BufType>
PipelineStepI* SetBufferStep<BufType>::Clone() const
{
    SetBufferStep* clone = new SetBufferStep<BufType>(*this);
    return clone;
}

template <class BufType>
void SetBufferStep<BufType>::ProcessStep(const VectorBufferTemplate<long long> indices,
                                        const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection) const
{
    UNUSED_PARAM(indices);
    UNUSED_PARAM(readCollection);

    if(!writeCollection.HasBuffer<BufType>(OutputBufferId) || mSetRule == EVERY_PROCESS)
    {
        writeCollection.AddBuffer(OutputBufferId, mBuffer);
    }
}