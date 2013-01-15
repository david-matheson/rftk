#pragma once

#include "BufferCollection.h"
#include "FeatureExtractorI.h"
#include "BestSplitI.h"
#include "SplitCriteriaI.h"
#include "NodeDataCollectorI.h"

class TrainConfigParams
{
public:
    TrainConfigParams(  std::vector<FeatureExtractorI*> featureExtractors,
                        NodeDataCollectorFactoryI* nodeDataCollectorFactory,
                        BestSplitI* bestSplit,
                        SplitCriteriaI* splitCriteria,
                        int numberOfTrees,
                        int maxNumberOfNodes);

    int GetIntParamsMaxDim();

    int GetFloatParamsMaxDim();

    int GetYDim();

    std::vector<FeatureExtractorI*> mFeatureExtractors;
    NodeDataCollectorFactoryI* mNodeDataCollectorFactory;
    BestSplitI* mBestSplit;
    SplitCriteriaI* mSplitCriteria;
    int mNumberOfTrees;
    int mMaxNumberOfNodes;
};

