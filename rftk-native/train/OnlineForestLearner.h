#pragma once

#include <set>
#include <utility>

#include "Forest.h"

#include "ActiveSplitNode.h"
#include "TrainConfigParams.h"
#include "OfflineSamplingParams.h"


class OnlineForestLearner
{
public:

    OnlineForestLearner( const TrainConfigParams& trainConfigParams );

    Forest GetForest() const;

    void Train(BufferCollection data, MatrixBufferInt indices );

private:
    TrainConfigParams mTrainConfigParams;
    Forest mForest;
    std::map< std::pair<int, int>, ActiveSplitNode* > mActiveNodes;
};

