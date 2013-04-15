#pragma once

#include <BufferCollection.h>
#include <BufferCollectionStack.h>

// ----------------------------------------------------------------------------
//
// PipelineStepI reads inputs from readCollection does some processing and
// writes the results to writeCollection
//
// ----------------------------------------------------------------------------

class PipelineStepI
{
public:
    virtual ~PipelineStepI() {}
    virtual void ProcessStep(   const Int64VectorBuffer indices,
                                const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const = 0;
    virtual PipelineStepI* Clone() const = 0;
};