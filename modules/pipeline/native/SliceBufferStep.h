#pragma once

#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Slice a buffer by a collection of row indices
//
// ----------------------------------------------------------------------------
template <class BufType, class IntType>
class SliceBufferStep: public PipelineStepI
{
public:
    SliceBufferStep(  const UniqueBufferId::BufferId& buffer,
                      const UniqueBufferId::BufferId& indices );
    virtual ~SliceBufferStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

    // Read only output buffer
    const UniqueBufferId::BufferId SlicedBufferId;
private:
    const UniqueBufferId::BufferId mBufferBufferId;
    const UniqueBufferId::BufferId mIndicesBufferId;
};


template <class BufType, class IntType>
SliceBufferStep<BufType, IntType>::SliceBufferStep(const UniqueBufferId::BufferId& buffer,
                                          const UniqueBufferId::BufferId& indices)
: SlicedBufferId(UniqueBufferId::GetBufferId("SlicedBuffer"))
, mBufferBufferId(buffer)
, mIndicesBufferId(indices)
{}

template <class BufType, class IntType>
SliceBufferStep<BufType, IntType>::~SliceBufferStep()
{}

template <class BufType, class IntType>
PipelineStepI* SliceBufferStep<BufType, IntType>::Clone() const
{
    SliceBufferStep<BufType, IntType>* clone = new SliceBufferStep<BufType, IntType>(*this);
    return clone;
}

template <class BufType, class IntType>
void SliceBufferStep<BufType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection) const
{
    const BufType& buffer = readCollection.GetBuffer<BufType>(mBufferBufferId);
    const VectorBufferTemplate<IntType>& indices = 
          readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId);
    const BufType& slicedBuffer = buffer.Slice(indices);
    writeCollection.AddBuffer<BufType>(SlicedBufferId, slicedBuffer);
}