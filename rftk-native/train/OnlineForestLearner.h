#pragma once

#include <set>
#include <utility>

#include "Forest.h"

#include "ActiveSplitNode.h"
#include "TrainConfigParams.h"
#include "OnlineSamplingParams.h"


class OnlineForestLearner
{
public:

    OnlineForestLearner( const TrainConfigParams& trainConfigParams );

    Forest GetForest() const;

    void Train(BufferCollection data, Int32VectorBuffer indices, OnlineSamplingParams samplingParams );

private:
    TrainConfigParams mTrainConfigParams;
    Forest mForest;
    std::map< std::pair<int, int>, ActiveSplitNode* > mActiveNodes;
};

