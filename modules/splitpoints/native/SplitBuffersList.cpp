#include "SplitBuffersList.h"

SplitBuffersList::SplitBuffersList(const std::vector<SplitBuffersI*>& bufferSplitters)
: mBufferSplitters()
{
    // Create a copy each buffer splitter
    for (std::vector<SplitBuffersI*>::const_iterator it = bufferSplitters.begin(); it != bufferSplitters.end(); ++it)
    {
        mBufferSplitters.push_back( (*it)->Clone() );
    }
}

SplitBuffersList::~SplitBuffersList()
{
    // Free each buffer splitter
    for (std::vector<SplitBuffersI*>::iterator it = mBufferSplitters.begin(); it != mBufferSplitters.end(); ++it)
    {
        delete (*it);
    }
}

void SplitBuffersList::SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                              int bestFeature,
                              int bestSplitpoint,
                              const BufferCollectionStack& readBuffers,
                              BufferCollection& leftBuffers, 
                              BufferCollection& rightBuffers) const
{
    for (std::vector<SplitBuffersI*>::const_iterator it = mBufferSplitters.begin(); it != mBufferSplitters.end(); ++it)
    {
        (*it)->SplitBuffers(splitSelectorBuffers,
                                bestFeature,
                                bestSplitpoint,
                                readBuffers,
                                leftBuffers,
                                rightBuffers);
    }
}

SplitBuffersI* SplitBuffersList::Clone() const
{
    SplitBuffersI* clone = new SplitBuffersList(mBufferSplitters);
    return clone;
}