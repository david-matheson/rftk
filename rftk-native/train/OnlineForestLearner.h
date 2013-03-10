#pragma once

#include <set>
#include <utility>
#include <vector>

#include "Forest.h"

#include "ActiveSplitNode.h"
#include "TrainConfigParams.h"
#include "OnlineSamplingParams.h"


class OnlineForestLearner
{
public:

    OnlineForestLearner( const TrainConfigParams& trainConfigParams, 
                          const OnlineSamplingParams& samplingParams, 
                          const unsigned int maxFrontierSize);
    ~OnlineForestLearner();

    Forest GetForest() const;

    void Train(BufferCollection data, Int32VectorBuffer indices );

private:
    TrainConfigParams mTrainConfigParams;
    OnlineSamplingParams mOnlineSamplingParams;
    unsigned int mMaxFrontierSize;

    Forest mForest;
    std::set< std::pair<int, int> > mQueuedFrontierLeaves;
    std::map< std::pair<int, int>, ActiveSplitNode* > mActiveFrontierLeaves;

    // Used for probability of a datapoint reaches a node
    std::vector<long long> mNumberOfDatapointsProcessedByTree;
    std::map< std::pair<int, int>, long long> mNumberOfDatapointsProcessedByTreeWhenNodeCreated; 
};

