#include "Pipeline.h"

Pipeline::Pipeline(const std::vector<PipelineStepI*>& steps)
: mSteps()
{
    // Create a copy each step
    for (std::vector<PipelineStepI*>::const_iterator it = steps.begin(); it != steps.end(); ++it)
    {
        mSteps.push_back( (*it)->Clone() );
    }
}

Pipeline::~Pipeline()
{
    // Free each step in the pipeline
    for (std::vector<PipelineStepI*>::iterator it = mSteps.begin(); it != mSteps.end(); ++it)
    {
        delete (*it);
    }
}

void Pipeline::ProcessStep( const BufferCollectionStack& readCollection,
                            BufferCollection& writeCollection) const
{
    for (std::vector<PipelineStepI*>::const_iterator it = mSteps.begin(); it != mSteps.end(); ++it)
    {
        (*it)->ProcessStep(readCollection, writeCollection);
    }
}

PipelineStepI* Pipeline::Clone() const
{
    Pipeline* clone = new Pipeline(mSteps);
    return clone;
}