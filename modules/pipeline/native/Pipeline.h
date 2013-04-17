#pragma once

#include <vector>
#include "PipelineStepI.h"

// ----------------------------------------------------------------------------
//
// Pipeline runs a sequence of PipelineSteps.  It is also a PipelineStepI so
// pipelines can be embedded as steps of pipelines.
// The pipeline owns a copy of all steps in the pipeline.
//
// ----------------------------------------------------------------------------
class Pipeline: public PipelineStepI
{
public:
    Pipeline(const std::vector<PipelineStepI*>& steps);
    virtual ~Pipeline();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

private:
    std::vector<PipelineStepI*> mSteps;
};

