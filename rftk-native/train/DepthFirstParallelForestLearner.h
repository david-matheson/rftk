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

    Forest Train(   BufferCollection data,
                    MatrixBufferInt indices,
                    const OfflineSamplingParams& samplingParams,
                    int numberOfJobs );

private:

    // void TrainTrees(    BufferCollection data,
    //                     MatrixBufferInt indices,
    //                     const OfflineSamplingParams& samplingParams,
    //                     int startIndex,
    //                     int offset,
    //                     Forest& forestOut );


    // void TrainTree( BufferCollection data,
    //                 MatrixBufferInt indices,
    //                 const OfflineSamplingParams& samplingParams,
    //                 Tree& treeOut);

    // void ProcessNode(   int nodeIndex, int treeDepth,
    //                     BufferCollection& data, const MatrixBufferInt& indices, Tree& treeOut);

    TrainConfigParams mTrainConfigParams;
};