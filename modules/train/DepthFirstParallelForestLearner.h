#pragma once

#include "buffers/MatrixBuffer.h"
#include "forest_data/Forest.h"

#include "TrainConfigParams.h"
#include "OfflineSamplingParams.h"
#include "ActiveSplitNode.h"

class DepthFirstParallelForestLearner
{
public:
    DepthFirstParallelForestLearner( const TrainConfigParams& trainConfigParams );

    Forest Train(   BufferCollection& data,
                    const Int32VectorBuffer& indices,
                    const OfflineSamplingParams& samplingParams,
                    int numberOfJobs );

private:
    TrainConfigParams mTrainConfigParams;
};