#pragma once

#include <vector>

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorI.h"



enum ACTIVE_NODE_STATE
{
    ACTIVE_NODE_STATE_MORE_DATA_REQUIRED,
    ACTIVE_NODE_STATE_READY_TO_SPLIT,
    ACTIVE_NODE_STATE_STOP,
};

// Subsampling could live in here along with a max cap
class NodeDataCollectorI
{
public:
    // Also copies/compacts weights, ys, etc
    virtual void Collect( const BufferCollection& data,
                          const MatrixBufferInt& sampleIndices,
                          const MatrixBufferFloat& featureValues ) {}

    // Includes feature values, weights, ys, etc
    virtual BufferCollection GetCollectedData() {}

    virtual int GetNumberOfCollectedSamples() { return 0; }
};

class NodeDataCollectorFactoryI
{
public:
    virtual NodeDataCollectorI* Create() { return NULL; }
};


class BestSplitI //Already exists
{
public:
    virtual void BestSplits( BufferCollection& data,
                            // const MatrixBufferInt& sampleIndices,
                            // const MatrixBufferFloat& featureValues, // contained in data (if needed)
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferInt& childCountsOut) {}
};

class SplitCriteriaI
{
public:
    virtual bool ShouldSplit(   int treeDepth,
                                int numberOfSamples
                                const MatrixBufferFloat& impurityValues, 
                                const MatrixBufferInt& childCounts,
                                int bestFeature ) { return false; }
};

class ActiveSplitNodeFeatureCandidate
{
public: 
    ActiveSplitNodeFeatureCandidats(    int index,
                                        FeatureExtractorI* featureExtractor,
                                        const NodeDataCollectorI* nodeDataCollector,
                                        const BestSplitI* bestSplitter )
    : mIndex(index)
    , mFeatureExtractor(featureExtractor)
    , mNodeDataCollector(nodeDataCollector)
    , mBestSplitter(bestSplitter)
    {
        //Init mIntParams and mFloatParams
        //setup mImpurities mThresholds, mChildCounts, mBestFeatureIndex
    }

    void ProcessData(    const BufferCollection& data,
                        const MatrixBufferInt& sampleIndices )
    {
        // Extract feature values
        MatrixBufferFloat featureValues;
        mFeatureExtractor->Extract(mIntParams, mFloatParams, featureValues)

        // Collect data
        mNodeDataCollector->Collect(data, sampleIndices, featureValues);

        // Calculate impurity
        mBestSplitter->BestSplits( mNodeDataCollector->GetCollectedData(),
                                   mImpurities,
                                   mThresholds,
                                   mChildCounts );      
    }

    void GetImpurity(   int startIndexOut, 
                        MatrixBufferFloat& impuritiesOut,
                        MatrixBufferFloat& thresholdsOut,
                        MatrixBufferInt& childCountsOut,
                        MatrixBufferInt& featureId )
    {
        // iteratre all values of mImpurities, mThresholds, mChildCounts
        // and populate outputs
    }

    int GetNumberFeatureCandidates()
    {
        return mIntParams.GetM();
    }

private:
    int mIndex;
    FeatureExtractorI* mFeatureExtractor;
    NodeDataCollectorI* mNodeDataCollector;
    BestSplitI* mBestSplitter;
    MatrixBufferInt mIntParams;
    MatrixBufferFloat mFloatParams;

    MatrixBufferFloat mImpurities;
    MatrixBufferFloat mThresholds;
    MatrixBufferInt mChildCounts;    
};

class FeatureIndex
{
    FeatureIndex()
    : mFeatureExtractorIndex(0)
    , mParamIndex(0)
    {}

    int mFeatureExtractorIndex;
    int mParamIndex;
};

class ActiveSplitNode
{
public:
    ActiveSplitNode(const std::vector<FeatureExtractorI> featureExtractors, 
                    const NodeDataCollectorFactoryI* nodeDataCollectorFactory,
                    const BestSplitI* bestSplit,
                    const SplitCriteriaI* splitCriteria, 
                    int treeDepth ) 
    : mShouldSplit(ACTIVE_NODE_STATE_MORE_DATA_REQUIRED)
    {
        for(int i = 0; i < mActiveSplitNodeFeatureCandidates.size(); i++)
        {
            //setup mActiveSplitNodeFeatureCandidates
            ActiveSplitNodeFeatureCandidate* a = ActiveSplitNodeFeatureCandidate(featureExtractors[i], 
                nodeDataCollectorFactory.Create(), bestSplit);
            mActiveSplitNodeFeatureCandidates.push_back(a);
    }

    //
    void ProcessData(    const BufferCollection& data,
                    const MatrixBufferInt& sampleIndices )
    {
        for(int i = 0; i < mActiveSplitNodeFeatureCandidates.size(); i++)
        {
            mActiveSplitNodeFeatureCandidates[i].ProcessData(data, sampleIndices);
        }

        int startIndex = 0
        for(int i = 0; i < mActiveSplitNodeFeatureCandidates.size(); i++)
        {
            mActiveSplitNodeFeatureCandidates[i].GetImpurity(startIndex, mImpurities, mThresholds, mChildCounts, mFeatureIds);
            startIndex += mActiveSplitNodeFeatureCandidates[i].GetNumberFeatureCandidates();
        }

        // Find max impurity and save mBestFeatureIndex
    }

    // Data has to be passed in because ProcessData may not keep the data
    void SplitIndices(  const BufferCollection& data,
                        FeatureIndex featureIndex,
                        const MatrixBufferInt& sampleIndices,
                        MatrixBufferInt& trueSampleIndicesOut,
                        MatrixBufferInt& falseSampleIndicesOut )
    {}

    FeatureIndex GetBestFeatureIndex()
    {
        return mBestFeatureIndex;
    }

private:
    std::vector<ActiveSplitNodeFeatureCandidate*> mActiveSplitNodeFeatureCandidates;

    MatrixBufferFloat mImpurities;
    MatrixBufferFloat mThresholds;
    MatrixBufferInt mChildCounts;  
    MatrixBufferInt mFeatureIds;  

    // Across all mActiveSplitNodeFeatureCandidates
    FeatureIndex mBestFeatureIndex;
    ACTIVE_NODE_STATE mShouldSplit;
};