#include <cstdio>

#include "TrainConfigParams.h"


TrainConfigParams::TrainConfigParams(  std::vector<FeatureExtractorI*> featureExtractors,
                    NodeDataCollectorFactoryI* nodeDataCollectorFactory,
                    BestSplitI* bestSplit,
                    SplitCriteriaI* splitCriteria,
                    int numberOfTrees,
                    int initialNumberOfNodes)
: mFeatureExtractors(0)
, mNodeDataCollectorFactory(nodeDataCollectorFactory->Clone())
, mBestSplit(bestSplit->Clone())
, mSplitCriteria(splitCriteria->Clone())
, mNumberOfTrees(numberOfTrees)
, mInitialNumberOfNodes(initialNumberOfNodes)
{
    for(unsigned int i=0; i<featureExtractors.size(); i++)
    {
        mFeatureExtractors.push_back( featureExtractors[i]->Clone() );
    }
}

TrainConfigParams::TrainConfigParams( const TrainConfigParams& other )
: mFeatureExtractors(0)
, mNodeDataCollectorFactory(other.mNodeDataCollectorFactory->Clone())
, mBestSplit(other.mBestSplit->Clone())
, mSplitCriteria(other.mSplitCriteria->Clone())
, mNumberOfTrees(other.mNumberOfTrees)
, mInitialNumberOfNodes(other.mInitialNumberOfNodes)
{
    for(unsigned int i=0; i<other.mFeatureExtractors.size(); i++)
    {
        mFeatureExtractors.push_back( other.mFeatureExtractors[i]->Clone() );
    }
}

TrainConfigParams& TrainConfigParams::operator=( const TrainConfigParams& rhs )
{
    Free();

    for(unsigned int i=0; i<rhs.mFeatureExtractors.size(); i++)
    {
        mFeatureExtractors.push_back( rhs.mFeatureExtractors[i]->Clone() );
    }

    mNodeDataCollectorFactory = rhs.mNodeDataCollectorFactory->Clone();
    mBestSplit = rhs.mBestSplit->Clone();
    mSplitCriteria = rhs.mSplitCriteria->Clone();
    mNumberOfTrees = rhs.mNumberOfTrees;
    mInitialNumberOfNodes = rhs.mInitialNumberOfNodes;
    return *this;
}

TrainConfigParams::~TrainConfigParams()
{
    Free();
}

void TrainConfigParams::Free()
{
    delete mNodeDataCollectorFactory;
    mNodeDataCollectorFactory = NULL;
    delete mBestSplit;
    mBestSplit = NULL;
    delete mSplitCriteria;
    mSplitCriteria = NULL;
    for(unsigned int i=0; i<mFeatureExtractors.size(); i++)
    {
        delete mFeatureExtractors[i];
        mFeatureExtractors[i] = NULL;

    }
    mFeatureExtractors.resize(0);
}

int TrainConfigParams::GetIntParamsMaxDim() const
{
    int maxDim = 0;
    for(unsigned int i=0; i<mFeatureExtractors.size(); i++)
    {
        maxDim = (maxDim >= mFeatureExtractors[i]->GetIntParamsDim()) ? maxDim : mFeatureExtractors[i]->GetIntParamsDim();
    }
    return maxDim;
}

int TrainConfigParams::GetFloatParamsMaxDim() const
{
    int maxDim = 0;
    for(unsigned int i=0; i<mFeatureExtractors.size(); i++)
    {
        maxDim = (maxDim >= mFeatureExtractors[i]->GetFloatParamsDim()) ? maxDim : mFeatureExtractors[i]->GetFloatParamsDim();
    }
    return maxDim;
}

int TrainConfigParams::GetYDim() const
{
    return mBestSplit->GetYDim();
}


