#include <vector>

#include "assert_util.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "SplitCriteriaI.h"
#include "ActiveSplitNode.h"



ActiveSplitNodeFeatureSet::ActiveSplitNodeFeatureSet(  const FeatureExtractorI* featureExtractor,
                                                        NodeDataCollectorI* nodeDataCollector,
                                                        const BestSplitI* bestSplitter )
: mFeatureExtractor(featureExtractor)
, mNodeDataCollector(nodeDataCollector)
, mBestSplitter(bestSplitter)
{
    const int numberOfFeatures = featureExtractor->GetNumberOfFeatures();
    mFloatParams = featureExtractor->CreateFloatParams(numberOfFeatures);
    mIntParams = featureExtractor->CreateIntParams(numberOfFeatures);
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());

    const int numberOfCandidateFeatures = mFloatParams.GetM();
    mImpurities = MatrixBufferFloat(numberOfCandidateFeatures, 1);
    mThresholds = MatrixBufferFloat(numberOfCandidateFeatures, 1);
    mChildCounts = MatrixBufferFloat(numberOfCandidateFeatures, 2);
    mLeftYs = MatrixBufferFloat(numberOfCandidateFeatures, bestSplitter->GetYDim() + 1);
    mRightYs = MatrixBufferFloat(numberOfCandidateFeatures, bestSplitter->GetYDim() + 1);
}

ActiveSplitNodeFeatureSet::~ActiveSplitNodeFeatureSet()
{
}

void ActiveSplitNodeFeatureSet::ProcessData(    BufferCollection& data,
                                                const MatrixBufferInt& sampleIndices )
{
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());
    ASSERT_ARG_DIM_2D(mImpurities.GetM(), mImpurities.GetN(), mThresholds.GetM(), mThresholds.GetN());
    ASSERT_ARG_DIM_1D(mImpurities.GetM(), mChildCounts.GetM());
    ASSERT_ARG_DIM_1D(mImpurities.GetN(), 1);
    ASSERT_ARG_DIM_1D(mThresholds.GetN(), 1);
    ASSERT_ARG_DIM_1D(mChildCounts.GetN(), 2);

    // Extract feature values
    MatrixBufferFloat featureValues;
    mFeatureExtractor->Extract(data, sampleIndices, mIntParams, mFloatParams, featureValues);

    // Collect data
    mNodeDataCollector->Collect(data, sampleIndices, featureValues);

    // Calculate impurity and child ys
    BufferCollection nodeBufferCollection = mNodeDataCollector->GetCollectedData();
    // Todo: make it clearer that we only calculate best splits when stats have been collected
    if( mNodeDataCollector->GetNumberOfCollectedSamples() > 0)
    {
        mBestSplitter->BestSplits( nodeBufferCollection,
                           mImpurities,
                           mThresholds,
                           mChildCounts,
                           mLeftYs,
                           mRightYs );
    }
}

void ActiveSplitNodeFeatureSet::WriteImpurity(  int groupId,
                                                int outStartIndex,
                                                MatrixBufferFloat& impuritiesOut,
                                                MatrixBufferFloat& thresholdsOut,
                                                MatrixBufferFloat& childCountsOut,
                                                MatrixBufferInt& featureIndicesOut  )
{
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());
    ASSERT_ARG_DIM_2D(mImpurities.GetM(), mImpurities.GetN(), mThresholds.GetM(), mThresholds.GetN());
    ASSERT_ARG_DIM_1D(mImpurities.GetM(), mChildCounts.GetM());
    ASSERT_ARG_DIM_1D(mImpurities.GetN(), 1);
    ASSERT_ARG_DIM_1D(mThresholds.GetN(), 1);
    ASSERT_ARG_DIM_1D(mChildCounts.GetN(), 2);

    for(int i=0; i<mFloatParams.GetM(); i++)
    {
        impuritiesOut.Set(outStartIndex+i,0, mImpurities.Get(i,0));
        thresholdsOut.Set(outStartIndex+i,0, mThresholds.Get(i,0));
        childCountsOut.Set(outStartIndex+i,0, mChildCounts.Get(i,0));
        childCountsOut.Set(outStartIndex+i,1, mChildCounts.Get(i,1));
        featureIndicesOut.Set(outStartIndex+i,0, groupId);
        featureIndicesOut.Set(outStartIndex+i,1, i);
    }
}

void ActiveSplitNodeFeatureSet::SplitIndices(   const int featureIndex,
                                                BufferCollection& data,
                                                const MatrixBufferInt& sampleIndices,
                                                MatrixBufferInt& leftSampleIndicesOut,
                                                MatrixBufferInt& rightSampleIndicesOut )
{
    // Extract feature values
    MatrixBufferFloat featureValues;
    mFeatureExtractor->Extract(data, sampleIndices, mIntParams, mFloatParams, featureValues);

    ASSERT_ARG_DIM_1D(sampleIndices.GetM(), featureValues.GetM());
    const float threshold = mThresholds.Get(featureIndex,0);

    std::vector<int> leftIndices;
    std::vector<int> rightIndices;

    for(int i=0; i<sampleIndices.GetM(); i++)
    {
        const float featureValue = featureValues.Get(i, featureIndex);
        const int sampleIndex = sampleIndices.Get(i, 0);
        std::vector<int>& leftOrRightIndices =
                    (featureValue >= threshold) ? leftIndices : rightIndices;
        leftOrRightIndices.push_back(sampleIndex);
    }

    leftSampleIndicesOut = MatrixBufferInt(&leftIndices[0], leftIndices.size(), 1);
    rightSampleIndicesOut = MatrixBufferInt(&rightIndices[0], rightIndices.size(), 1);
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
    // printf("ActiveSplitNodeFeatureSet::WriteToTree intparams\n");

    intParamsOut.Set(treeNodeIndex, 0, mFeatureExtractor->GetUID());
    for(int c = 1; c < mIntParams.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        intParamsOut.Set(treeNodeIndex, c, mIntParams.Get(index, c));
    }

    // printf("ActiveSplitNodeFeatureSet::WriteToTree floatparms\n");

    floatParamsOut.Set(treeNodeIndex, 0, mThresholds.Get(index, 0));
    for(int c = 1; c < mFloatParams.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        floatParamsOut.Set(treeNodeIndex, c, mFloatParams.Get(index, c));
    }

    // printf("ActiveSplitNodeFeatureSet::WriteToTree left leafs\n");

    for(int c = 0; c < mLeftYs.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        leftYsOut.Set(leftTreeNodeIndex, c, mLeftYs.Get(index, c));
    }

    // printf("ActiveSplitNodeFeatureSet::WriteToTree right leafs\n");

    for(int c = 0; c < mRightYs.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        rightYsOut.Set(rightTreeNodeIndex, c, mRightYs.Get(index, c));
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
    int numberOfFeatureCandidates = 0;
    for(int i = 0; i < featureExtractors.size(); i++)
    {
        //setup mActiveSplitNodeFeatureSets
        ActiveSplitNodeFeatureSet a = ActiveSplitNodeFeatureSet(   featureExtractors[i],
                                                                    nodeDataCollectorFactory->Create(),
                                                                    bestSplit);
        mActiveSplitNodeFeatureSets.push_back(a);
        numberOfFeatureCandidates += a.GetNumberFeatureCandidates();
    }
    mImpurities = MatrixBufferFloat(numberOfFeatureCandidates, 1);
    mThresholds = MatrixBufferFloat(numberOfFeatureCandidates, 1);
    mChildCounts = MatrixBufferFloat(numberOfFeatureCandidates, 2);
    mFeatureIndices = MatrixBufferInt(numberOfFeatureCandidates, 2);
}

ActiveSplitNode::~ActiveSplitNode()
{
}


void ActiveSplitNode::ProcessData(  BufferCollection& data,
                                    const MatrixBufferInt& sampleIndices )
{
    // printf("ActiveSplitNode::ProcessData\n");
    for(int i = 0; i < mActiveSplitNodeFeatureSets.size(); i++)
    {
        mActiveSplitNodeFeatureSets[i].ProcessData(data, sampleIndices);
    }

    // printf("ActiveSplitNode::ProcessData WriteImpurity\n");
    int startIndex = 0;
    for(int i = 0; i < mActiveSplitNodeFeatureSets.size(); i++)
    {
        mActiveSplitNodeFeatureSets[i].WriteImpurity(startIndex, i-startIndex, mImpurities, mThresholds, mChildCounts, mFeatureIndices);
        startIndex += mActiveSplitNodeFeatureSets[i].GetNumberFeatureCandidates();
    }

    // printf("ActiveSplitNode::ProcessData BestSplit\n");
    mBestFeatureIndex = mSplitCriteria->BestSplit(mTreeDepth, mImpurities, mChildCounts);
    mShouldSplit = mSplitCriteria->ShouldSplit(mTreeDepth, mImpurities, mChildCounts);
    // printf("ActiveSplitNode::ProcessData BestSplit Result %d %d\n", mBestFeatureIndex, mShouldSplit);
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
    ASSERT_VALID_RANGE(mBestFeatureIndex, 0, mImpurities.GetM())

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
void ActiveSplitNode::SplitIndices(  BufferCollection& data,
                    const MatrixBufferInt& sampleIndices,
                    MatrixBufferInt& leftSampleIndicesOut,
                    MatrixBufferInt& rightSampleIndicesOut )
{
    ASSERT_VALID_RANGE(mBestFeatureIndex, 0, mImpurities.GetM())

    const int featureSetIndex = mFeatureIndices.Get(mBestFeatureIndex, 0);
    const int featureSetOffsetIndex = mFeatureIndices.Get(mBestFeatureIndex, 1);
    mActiveSplitNodeFeatureSets[featureSetIndex].SplitIndices(featureSetOffsetIndex, data, sampleIndices,
                                                                leftSampleIndicesOut, rightSampleIndicesOut);
}

