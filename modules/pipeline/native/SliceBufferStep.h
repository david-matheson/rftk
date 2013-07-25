#pragma once

#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Slice a buffer by a collection of row indices
//
// ----------------------------------------------------------------------------
template <class BufferTypes, class DataBufferType>
class SliceBufferStep: public PipelineStepI
{
public:
    SliceBufferStep(  const BufferId& buffer,
                      const BufferId& indices );
    virtual ~SliceBufferStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    // Read only output buffer
    const BufferId SlicedBufferId;
private:
    const BufferId mBufferBufferId;
    const BufferId mIndicesBufferId;
};


template <class BufferTypes, class DataBufferType>
SliceBufferStep<BufferTypes, DataBufferType>::SliceBufferStep(const BufferId& buffer,
                                                                const BufferId& indices)
: SlicedBufferId(buffer+GetBufferId("-Sliced"))
, mBufferBufferId(buffer)
, mIndicesBufferId(indices)
{}

template <class BufferTypes, class DataBufferType>
SliceBufferStep<BufferTypes, DataBufferType>::~SliceBufferStep()
{}

template <class BufferTypes, class DataBufferType>
PipelineStepI* SliceBufferStep<BufferTypes, DataBufferType>::Clone() const
{
    SliceBufferStep<BufferTypes, DataBufferType>* clone = new SliceBufferStep<BufferTypes, DataBufferType>(*this);
    return clone;
}

template <class BufferTypes, class DataBufferType>
void SliceBufferStep<BufferTypes, DataBufferType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection,
                                                                boost::mt19937& gen) const
{
    UNUSED_PARAM(gen);
    const DataBufferType& buffer = readCollection.GetBuffer<DataBufferType>(mBufferBufferId);
    const VectorBufferTemplate<typename BufferTypes::Index>& indices =
          readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId);
    const DataBufferType& slicedBuffer = buffer.Slice(indices);
    writeCollection.AddBuffer<DataBufferType>(SlicedBufferId, slicedBuffer);
}