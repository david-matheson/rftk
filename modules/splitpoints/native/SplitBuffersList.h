#pragma once

#include <vector>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "SplitSelectorBuffers.h"
#include "SplitBuffersI.h"

class SplitBuffersList : public SplitBuffersI
{
public:
    SplitBuffersList(const std::vector<SplitBuffersI*>& bufferSplitters);
    virtual ~SplitBuffersList();

    virtual void SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                              int bestFeature,
                              int bestSplitpoint,
                              const BufferCollectionStack& readBuffers,
                              BufferCollection& leftBuffers, 
                              BufferCollection& rightBuffers) const;

    virtual SplitBuffersI* Clone() const;

private:
    std::vector<SplitBuffersI*> mBufferSplitters;
};


