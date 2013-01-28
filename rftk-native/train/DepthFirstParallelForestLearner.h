#pragma once

#include "MatrixBuffer.h"
#include "Forest.h"

#include "TrainConfigParams.h"
#include "OfflineSamplingParams.h"
#include "ActiveSplitNode.h"

class DepthFirstParallelForestLearner
{
public:
    DepthFirstParallelForestLearner( const TrainConfigParams& trainConfigParams );

    Forest Train(   BufferCollection& data,
                    const Int32MatrixBuffer& indices,
                    const OfflineSamplingParams& samplingParams,
                    int numberOfJobs );

private:
    TrainConfigParams mTrainConfigParams;
};