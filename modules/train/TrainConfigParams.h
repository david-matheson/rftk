#pragma once

#include <tr1/memory>
#include <vector>

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
                        int initialNumberOfNodes = 32);
    TrainConfigParams( const TrainConfigParams& other );
    TrainConfigParams & operator=( const TrainConfigParams& rhs );
    ~TrainConfigParams();
    void Free();

    int GetIntParamsMaxDim() const;
    int GetFloatParamsMaxDim() const;

    int GetYDim() const;

    std::vector< FeatureExtractorI*> mFeatureExtractors;
    NodeDataCollectorFactoryI* mNodeDataCollectorFactory;
    BestSplitI* mBestSplit;
    SplitCriteriaI* mSplitCriteria;
    int mNumberOfTrees;
    int mInitialNumberOfNodes;

};

