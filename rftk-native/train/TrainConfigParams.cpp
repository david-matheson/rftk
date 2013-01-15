#include "TrainConfigParams.h"


TrainConfigParams::TrainConfigParams(  std::vector<FeatureExtractorI*> featureExtractors,
                    NodeDataCollectorFactoryI* nodeDataCollectorFactory,
                    BestSplitI* bestSplit,
                    SplitCriteriaI* splitCriteria,
                    int numberOfTrees,
                    int maxNumberOfNodes)
: mFeatureExtractors(featureExtractors)
, mNodeDataCollectorFactory(nodeDataCollectorFactory)
, mBestSplit(bestSplit)
, mSplitCriteria(splitCriteria)
, mNumberOfTrees(numberOfTrees)
, mMaxNumberOfNodes(maxNumberOfNodes)
{}

int TrainConfigParams::GetIntParamsMaxDim()
{
    int maxDim = 0;
    for(int i=0; i<mFeatureExtractors.size(); i++)
    {
        maxDim = (maxDim >= mFeatureExtractors[i]->GetIntParamsDim()) ? maxDim : mFeatureExtractors[i]->GetIntParamsDim();
    }
    return maxDim;
}

int TrainConfigParams::GetFloatParamsMaxDim()
{
    int maxDim = 0;
    for(int i=0; i<mFeatureExtractors.size(); i++)
    {
        maxDim = (maxDim >= mFeatureExtractors[i]->GetFloatParamsDim()) ? maxDim : mFeatureExtractors[i]->GetFloatParamsDim();
    }
    return maxDim;
}

int TrainConfigParams::GetYDim()
{
    return mBestSplit->GetYDim();
}


