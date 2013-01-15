#include <vector>

#include "assert_util.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorI.h"
#include "ActiveSplitNode.h"



ActiveSplitNodeFeatureSet::ActiveSplitNodeFeatureSet(  const FeatureExtractorI* featureExtractor,
                                                        NodeDataCollectorI* nodeDataCollector,
                                                        const BestSplitI* bestSplitter )
: mFeatureExtractor(featureExtractor)
, mNodeDataCollector(nodeDataCollector)
, mBestSplitter(bestSplitter)
{
    mFloatParams = featureExtractor->CreateFloatParams();
    mIntParams = featureExtractor->CreateIntParams();
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());

    const int numberOfCandidateFeatures = mFloatParams.GetM();
    mImpurities = MatrixBufferFloat(numberOfCandidateFeatures, 1);
    mThresholds = MatrixBufferFloat(numberOfCandidateFeatures, 1);
    mChildCounts = MatrixBufferInt(numberOfCandidateFeatures, 2);
    mLeftYs = MatrixBufferFloat(numberOfCandidateFeatures, bestSplitter->GetYDim());
    mRightYs = MatrixBufferFloat(numberOfCandidateFeatures, bestSplitter->GetYDim());
}

ActiveSplitNodeFeatureSet::~ActiveSplitNodeFeatureSet()
{
    delete mNodeDataCollector;
}

void ActiveSplitNodeFeatureSet::ProcessData(    const BufferCollection& data,
                                                const MatrixBufferInt& sampleIndices )
{
    // Extract feature values
    MatrixBufferFloat featureValues;
    mFeatureExtractor->Extract(data, sampleIndices, mIntParams, mFloatParams, featureValues);

    // Collect data
    mNodeDataCollector->Collect(data, sampleIndices, featureValues);

    // Calculate impurity and
    BufferCollection nodeBufferCollection = mNodeDataCollector->GetCollectedData();
    mBestSplitter->BestSplits( nodeBufferCollection,
                               mImpurities,
                               mThresholds,
                               mChildCounts,
                               mLeftYs,
                               mRightYs );
}


void ActiveSplitNodeFeatureSet::WriteToTree(int index,
                                            const int treeNodeIndex,
                                            MatrixBufferFloat& floatParamsOut,
                                            MatrixBufferInt& intParamsOut,
                                            const int leftTreeNodeIndex,
                                            MatrixBufferFloat& leftYsOut,
                                            const int rightTreeNodeIndex,
                                            MatrixBufferFloat& rightYsOut )
{
    intParamsOut.Set(treeNodeIndex, 0, mFeatureExtractor->GetUID());
    for(int c = 1; c < mIntParams.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        intParamsOut.Set(treeNodeIndex, c, mIntParams.Get(index, c));
    }

    floatParamsOut.Set(treeNodeIndex, 0, mThresholds.Get(index, 0));
    for(int c = 1; c < mFloatParams.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        floatParamsOut.Set(treeNodeIndex, c, mFloatParams.Get(index, c));
    }

    for(int c = 0; c < mLeftYs.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        leftYsOut.Set(leftTreeNodeIndex, c, mLeftYs.Get(index, c));
    }

    for(int c = 0; c < mRightYs.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        rightYsOut.Set(leftTreeNodeIndex, c, mRightYs.Get(index, c));
    }
}

void ActiveSplitNodeFeatureSet::WriteImpurity(  int groupId,
                                                int outStartIndex,
                                                MatrixBufferFloat& impuritiesOut,
                                                MatrixBufferFloat& thresholdsOut,
                                                MatrixBufferInt& childCountsOut,
                                                MatrixBufferInt& featureIndicesOut  )
{
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());
    ASSERT_ARG_DIM_2D(mImpurities.GetM(), mImpurities.GetN(), mThresholds.GetM(), mThresholds.GetN());
    ASSERT_ARG_DIM_1D(mImpurities.GetM(), mChildCounts.GetM());
    ASSERT_ARG_DIM_1D(mImpurities.GetN(), 2);

    for(int i=0; i<mFloatParams.GetM(); i++)
    {
        impuritiesOut.Set(outStartIndex+i,0, mImpurities.Get(i,0));
        thresholdsOut.Set(outStartIndex+i,0, mThresholds.Get(i,0));
        childCountsOut.Set(outStartIndex+i,0, mThresholds.Get(i,0));
        childCountsOut.Set(outStartIndex+i,1, mThresholds.Get(i,1));
        featureIndicesOut.Set(outStartIndex+i,0, groupId);
        featureIndicesOut.Set(outStartIndex+i,0, i);
    }
}


ActiveSplitNode::ActiveSplitNode(const std::vector<FeatureExtractorI*> featureExtractors,
                const NodeDataCollectorFactoryI* nodeDataCollectorFactory,
                const BestSplitI* bestSplit,
                const SplitCriteriaI* splitCriteria,
                const int treeDepth )
: mSplitCriteria(splitCriteria)
, mTreeDepth(treeDepth)
, mBestFeatureIndex(-1)
, mShouldSplit(SPLT_CRITERIA_MORE_DATA_REQUIRED)
{
    for(int i = 0; i < featureExtractors.size(); i++)
    {
        //setup mActiveSplitNodeFeatureSets
        ActiveSplitNodeFeatureSet a = ActiveSplitNodeFeatureSet(   featureExtractors[i],
                                                                    nodeDataCollectorFactory->Create(),
                                                                    bestSplit);
        mActiveSplitNodeFeatureSets.push_back(a);
    }
}

ActiveSplitNode::~ActiveSplitNode()
{
}


void ActiveSplitNode::ProcessData(  const BufferCollection& data,
                                    const MatrixBufferInt& sampleIndices )
{
    for(int i = 0; i < mActiveSplitNodeFeatureSets.size(); i++)
    {
        mActiveSplitNodeFeatureSets[i].ProcessData(data, sampleIndices);
    }

    int startIndex = 0;
    for(int i = 0; i < mActiveSplitNodeFeatureSets.size(); i++)
    {
        mActiveSplitNodeFeatureSets[i].WriteImpurity(startIndex, i, mImpurities, mThresholds, mChildCounts, mFeatureIndices);
        startIndex += mActiveSplitNodeFeatureSets[i].GetNumberFeatureCandidates();
    }

    // Find max impurity and save mBestFeatureIndex

}


void ActiveSplitNode::WriteToTree(  const int treeNodeIndex,
                                    MatrixBufferInt& paths,
                                    MatrixBufferFloat& floatParams,
                                    MatrixBufferInt& intParams,
                                    MatrixBufferInt& depth,
                                    const int leftTreeNodeIndex,
                                    MatrixBufferFloat& leftYs,
                                    const int rightTreeNodeIndex,
                                    MatrixBufferFloat& rightYs )
{
    paths.Set(treeNodeIndex, 0, leftTreeNodeIndex);
    paths.Set(treeNodeIndex, 1, rightTreeNodeIndex);
    depth.Set(treeNodeIndex, 0, mTreeDepth);

    const int featureSetIndex = mFeatureIndices.Get(mBestFeatureIndex, 0);
    const int featureSetOffsetIndex = mFeatureIndices.Get(mBestFeatureIndex, 1);
    mActiveSplitNodeFeatureSets[featureSetIndex].WriteToTree( featureSetOffsetIndex,
                                                                treeNodeIndex,
                                                                floatParams,
                                                                intParams,
                                                                leftTreeNodeIndex,
                                                                leftYs,
                                                                rightTreeNodeIndex,
                                                                rightYs);
}

// Data has to be passed in because ProcessData may not keep the data
void ActiveSplitNode::SplitIndices(  const BufferCollection& data,
                    const MatrixBufferInt& sampleIndices,
                    MatrixBufferInt& leftSampleIndicesOut,
                    MatrixBufferInt& rightSampleIndicesOut )
{
    const int featureSetIndex = mFeatureIndices.Get(mBestFeatureIndex, 0);
    const int featureSetOffsetIndex = mFeatureIndices.Get(mBestFeatureIndex, 1);
    // mActiveSplitNodeFeatureSets[featureSetIndex].SplitIndices(featureSetOffsetIndex, data, sampleIndices,
    //                                                             trueSampleIndicesOut, falseSampleIndicesOut);
}

