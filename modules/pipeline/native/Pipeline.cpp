#include "BufferCollectionUtils.h"
#include "Pipeline.h"

Pipeline::Pipeline(const std::vector<PipelineStepI*>& steps)
: PipelineStepI("Pipeline")
, mSteps()
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
                            BufferCollection& writeCollection,
                            boost::mt19937& gen,
                            BufferCollection& extraInfo, int nodeIndex) const
{
    TimeLogger totalPipeline(extraInfo, "Pipeline");
    TimeLogger perNode(extraInfo, nodeIndex);
    for (std::vector<PipelineStepI*>::const_iterator it = mSteps.begin(); it != mSteps.end(); ++it)
    {
        TimeLogger perStep(extraInfo, (*it)->GetName());
        (*it)->ProcessStep(readCollection, writeCollection, gen, extraInfo, nodeIndex);
    }
}

PipelineStepI* Pipeline::Clone() const
{
    Pipeline* clone = new Pipeline(mSteps);
    return clone;
}