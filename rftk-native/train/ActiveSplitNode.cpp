#include <vector>
#include <cstdio>

#include "assert_util.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "SplitCriteriaI.h"
#include "ActiveSplitNode.h"



ActiveSplitNodeFeatureSet::ActiveSplitNodeFeatureSet(  const FeatureExtractorI* featureExtractor,
                                                        NodeDataCollectorI* nodeDataCollector,
                                                        const BestSplitI* bestSplitter,
                                                        const int evalSplitPeriod )
: mFeatureExtractor(featureExtractor)
, mBestSplitter(bestSplitter)
, mNodeDataCollector(nodeDataCollector)
, mEvalSplitPeriod(evalSplitPeriod)
, mNumberSamplesToEvalSplit(evalSplitPeriod)
{
    const int numberOfFeatures = featureExtractor->GetNumberOfFeatures();
    mFloatParams = featureExtractor->CreateFloatParams(numberOfFeatures);
    mIntParams = featureExtractor->CreateIntParams(numberOfFeatures);
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());

    const int numberOfCandidateFeatures = mFloatParams.GetM();
    mImpurities = Float32VectorBuffer(numberOfCandidateFeatures);
    mThresholds = Float32VectorBuffer(numberOfCandidateFeatures);
    mChildCounts = Float32MatrixBuffer(numberOfCandidateFeatures, 2);
    mLeftYs = Float32MatrixBuffer(numberOfCandidateFeatures, bestSplitter->GetYDim() + 1);
    mRightYs = Float32MatrixBuffer(numberOfCandidateFeatures, bestSplitter->GetYDim() + 1);
}

ActiveSplitNodeFeatureSet::~ActiveSplitNodeFeatureSet()
{
}

void ActiveSplitNodeFeatureSet::ProcessData(    const BufferCollection& data,
                                                const Int32VectorBuffer& sampleIndices,
                                                boost::mt19937& gen,
                                                const int minSamples )
{
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());
    ASSERT_ARG_DIM_1D(mImpurities.GetN(), mThresholds.GetN());
    ASSERT_ARG_DIM_1D(mImpurities.GetN(), mChildCounts.GetM());
    ASSERT_ARG_DIM_1D(mChildCounts.GetN(), 2);

    // Extract feature values
    Float32MatrixBuffer featureValues;
    mFeatureExtractor->Extract(data, sampleIndices, mIntParams, mFloatParams, featureValues);

    // Collect data
    mNodeDataCollector->Collect(data, sampleIndices, featureValues, gen);

    // Calculate impurity and child ys
    mNumberSamplesToEvalSplit -= sampleIndices.GetN();
    const BufferCollection& nodeBufferCollection = mNodeDataCollector->GetCollectedData();
    // Todo: make it clearer that we only calculate best splits when stats have been collected
    if( mNumberSamplesToEvalSplit <= 0 && mNodeDataCollector->GetNumberOfCollectedSamples() > minSamples)
    {
        mBestSplitter->BestSplits( nodeBufferCollection,
                           mImpurities,
                           mThresholds,
                           mChildCounts,
                           mLeftYs,
                           mRightYs );
        mNumberSamplesToEvalSplit = mEvalSplitPeriod;
    }
}

void ActiveSplitNodeFeatureSet::WriteImpurity(  int groupId,
                                                int outStartIndex,
                                                Float32VectorBuffer& impuritiesOut,
                                                Float32VectorBuffer& thresholdsOut,
                                                Float32MatrixBuffer& childCountsOut,
                                                Int32MatrixBuffer& featureIndicesOut  )
{
    ASSERT_ARG_DIM_1D(mFloatParams.GetM(), mIntParams.GetM());
    ASSERT_ARG_DIM_1D(mImpurities.GetN(), mThresholds.GetN());
    ASSERT_ARG_DIM_1D(mImpurities.GetN(), mChildCounts.GetM());
    ASSERT_ARG_DIM_1D(mChildCounts.GetN(), 2);

    for(int i=0; i<mFloatParams.GetM(); i++)
    {
        impuritiesOut.Set(outStartIndex+i, mImpurities.Get(i));
        thresholdsOut.Set(outStartIndex+i, mThresholds.Get(i));
        childCountsOut.Set(outStartIndex+i,0, mChildCounts.Get(i,0));
        childCountsOut.Set(outStartIndex+i,1, mChildCounts.Get(i,1));
        featureIndicesOut.Set(outStartIndex+i,0, groupId);
        featureIndicesOut.Set(outStartIndex+i,1, i);
    }
}

void ActiveSplitNodeFeatureSet::SplitIndices(   const int featureIndex,
                                                const BufferCollection& data,
                                                const Int32VectorBuffer& sampleIndices,
                                                Int32VectorBuffer& leftSampleIndicesOut,
                                                Int32VectorBuffer& rightSampleIndicesOut )
{
    // Extract feature values
    Float32MatrixBuffer featureValues;
    mFeatureExtractor->Extract(data, sampleIndices, mIntParams.SliceRow(featureIndex), mFloatParams.SliceRow(featureIndex), featureValues);

    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), featureValues.GetM())
    ASSERT_ARG_DIM_1D(featureValues.GetN(), 1)

    const float threshold = mThresholds.Get(featureIndex);
    std::vector<int> leftIndices;
    std::vector<int> rightIndices;

    for(int i=0; i<sampleIndices.GetN(); i++)
    {
        const float featureValue = featureValues.Get(i, 0);
        const int sampleIndex = sampleIndices.Get(i);
        if( featureValue >= threshold )
        {
            leftIndices.push_back(sampleIndex);
        }
        else
        {
            rightIndices.push_back(sampleIndex);
        }
    }

    leftSampleIndicesOut = Int32VectorBuffer(&leftIndices[0], leftIndices.size());
    rightSampleIndicesOut = Int32VectorBuffer(&rightIndices[0], rightIndices.size());
}

void ActiveSplitNodeFeatureSet::WriteToTree(int index,
                                            const int treeNodeIndex,
                                            const int leftTreeNodeIndex,
                                            const int rightTreeNodeIndex,
                                            Float32MatrixBuffer& floatParamsOut,
                                            Int32MatrixBuffer& intParamsOut,
                                            Float32VectorBuffer& countsOut,
                                            Float32MatrixBuffer& ysOut )
{
    intParamsOut.Set(treeNodeIndex, 0, mFeatureExtractor->GetUID());
    for(int c = 1; c < mIntParams.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        intParamsOut.Set(treeNodeIndex, c, mIntParams.Get(index, c));
    }

    floatParamsOut.Set(treeNodeIndex, 0, mThresholds.Get(index));
    for(int c = 1; c < mFloatParams.GetN(); c++)
    {
        floatParamsOut.Set(treeNodeIndex, c, mFloatParams.Get(index, c));
    }

    countsOut.Set(leftTreeNodeIndex, mChildCounts.Get(index, 0));
    countsOut.Set(rightTreeNodeIndex, mChildCounts.Get(index, 1));

    for(int c = 0; c < mLeftYs.GetN(); c++)
    {
        // Move slice copying functionality into MatrixBuffer
        ysOut.Set(leftTreeNodeIndex, c, mLeftYs.Get(index, c));
    }

    for(int c = 0; c < mRightYs.GetN(); c++)
    {
        ysOut.Set(rightTreeNodeIndex, c, mRightYs.Get(index, c));
    }
}

ActiveSplitNode::ActiveSplitNode(const std::vector<FeatureExtractorI*> featureExtractors,
                const NodeDataCollectorFactoryI* nodeDataCollectorFactory,
                const BestSplitI* bestSplit,
                const SplitCriteriaI* splitCriteria,
                const int treeDepth,
                const int evalSplitPeriod )
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
                                                                    bestSplit,
                                                                    evalSplitPeriod);
        mActiveSplitNodeFeatureSets.push_back(a);
        numberOfFeatureCandidates += a.GetNumberFeatureCandidates();
    }
    mImpurities = Float32VectorBuffer(numberOfFeatureCandidates);
    mThresholds = Float32VectorBuffer(numberOfFeatureCandidates);
    mChildCounts = Float32MatrixBuffer(numberOfFeatureCandidates, 2);
    mFeatureIndices = Int32MatrixBuffer(numberOfFeatureCandidates, 2);
}

ActiveSplitNode::~ActiveSplitNode()
{
}


void ActiveSplitNode::ProcessData(  const BufferCollection& data,
                                    const Int32VectorBuffer& sampleIndices,
                                    boost::mt19937& gen )
{
    // printf("ActiveSplitNode::ProcessData\n");
    for(int i = 0; i < mActiveSplitNodeFeatureSets.size(); i++)
    {
        mActiveSplitNodeFeatureSets[i].ProcessData(data, sampleIndices, gen, mSplitCriteria->MinTotalSamples(mTreeDepth));
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
                                    const int leftTreeNodeIndex,
                                    const int rightTreeNodeIndex,                                      
                                    Int32MatrixBuffer& paths,
                                    Float32MatrixBuffer& floatParams,
                                    Int32MatrixBuffer& intParams,
                                    Int32VectorBuffer& depth,
                                    Float32VectorBuffer& counts,
                                    Float32MatrixBuffer& ys  )
{
    ASSERT_VALID_RANGE(mBestFeatureIndex, 0, mImpurities.GetN())

    paths.Set(treeNodeIndex, 0, leftTreeNodeIndex);
    paths.Set(treeNodeIndex, 1, rightTreeNodeIndex);
    depth.Set(treeNodeIndex, mTreeDepth);
    depth.Set(leftTreeNodeIndex, mTreeDepth+1);
    depth.Set(rightTreeNodeIndex, mTreeDepth+1);

    const int featureSetIndex = mFeatureIndices.Get(mBestFeatureIndex, 0);
    const int featureSetOffsetIndex = mFeatureIndices.Get(mBestFeatureIndex, 1);

    mActiveSplitNodeFeatureSets[featureSetIndex].WriteToTree( featureSetOffsetIndex,
                                                                treeNodeIndex,
                                                                leftTreeNodeIndex,
                                                                rightTreeNodeIndex,
                                                                floatParams,
                                                                intParams,
                                                                counts,
                                                                ys);
}

// Data has to be passed in because ProcessData may not keep the data
void ActiveSplitNode::SplitIndices(  const BufferCollection& data,
                    const Int32VectorBuffer& sampleIndices,
                    Int32VectorBuffer& leftSampleIndicesOut,
                    Int32VectorBuffer& rightSampleIndicesOut )
{
    ASSERT_VALID_RANGE(mBestFeatureIndex, 0, mImpurities.GetN())

    const int featureSetIndex = mFeatureIndices.Get(mBestFeatureIndex, 0);
    const int featureSetOffsetIndex = mFeatureIndices.Get(mBestFeatureIndex, 1);
    mActiveSplitNodeFeatureSets[featureSetIndex].SplitIndices(featureSetOffsetIndex, data, sampleIndices,
                                                                leftSampleIndicesOut, rightSampleIndicesOut);
}

