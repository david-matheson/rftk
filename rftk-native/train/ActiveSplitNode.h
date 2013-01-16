#pragma once

#include <vector>

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorI.h"
#include "NodeDataCollectorI.h"
#include "BestSplitI.h"
#include "SplitCriteriaI.h"




// class BestSplitI //Already exists
// {
// public:
//     virtual int Ydim() { return 1; }

//     virtual void BestSplits( BufferCollection& data,
//                             // const MatrixBufferInt& sampleIndices,
//                             // const MatrixBufferFloat& featureValues, // contained in data (if needed)
//                             MatrixBufferFloat& impurityOut,
//                             MatrixBufferFloat& thresholdOut,
//                             MatrixBufferInt& childCountsOut,
//                             MatrixBufferFloat& leftYsOut,
//                             MatrixBufferFloat& rightYsOut) {}
// };



class ActiveSplitNodeFeatureSet
{
public:
    ActiveSplitNodeFeatureSet(  const FeatureExtractorI* featureExtractor,
                                NodeDataCollectorI* nodeDataCollector,
                                const BestSplitI* bestSplitter );

    ~ActiveSplitNodeFeatureSet();

    void ProcessData(    const BufferCollection& data,
                        const MatrixBufferInt& sampleIndices );

    void WriteImpurity( int groupId,
                        int outStartIndex,
                        MatrixBufferFloat& impuritiesOut,
                        MatrixBufferFloat& thresholdsOut,
                        MatrixBufferFloat& childCountsOut,
                        MatrixBufferInt& featureIndicesOut );

    void SplitIndices(  const int featureIndex,
                        const BufferCollection& data,
                        const MatrixBufferInt& sampleIndices,
                        MatrixBufferInt& leftSampleIndicesOut,
                        MatrixBufferInt& rightSampleIndicesOut );

    void WriteToTree(   int index,
                        const int treeNodeIndex,
                        MatrixBufferFloat& floatParamsOut,
                        MatrixBufferInt& intParamsOut,
                        const int leftTreeNodeIndex,
                        MatrixBufferFloat& leftYsOut,
                        const int rightTreeNodeIndex,
                        MatrixBufferFloat& rightYsOut );

    int GetNumberFeatureCandidates() const { return mIntParams.GetM(); }

private:
    // Passed in (not owned)
    const FeatureExtractorI* mFeatureExtractor;
    const BestSplitI* mBestSplitter;

    // Passed in but owned by this class
    NodeDataCollectorI* mNodeDataCollector;

    // Created on construction
    MatrixBufferInt mIntParams;
    MatrixBufferFloat mFloatParams;

    // Updated everytime ProcessData is called
    MatrixBufferFloat mImpurities;
    MatrixBufferFloat mThresholds;
    MatrixBufferFloat mChildCounts;
    MatrixBufferFloat mLeftYs;
    MatrixBufferFloat mRightYs;
};


class ActiveSplitNode
{
public:
    ActiveSplitNode(const std::vector<FeatureExtractorI*> featureExtractors,
                    const NodeDataCollectorFactoryI* nodeDataCollectorFactory,
                    const BestSplitI* bestSplit,
                    const SplitCriteriaI* splitCriteria,
                    const int treeDepth );

    virtual ~ActiveSplitNode();

    void ProcessData(   const BufferCollection& data,
                        const MatrixBufferInt& sampleIndices );

    SPLT_CRITERIA ShouldSplit() { return mShouldSplit; }

    void WriteToTree(   const int treeNodeIndex,
                        MatrixBufferInt& paths,
                        MatrixBufferFloat& floatParams,
                        MatrixBufferInt& intParams,
                        MatrixBufferInt& depth,
                        const int leftTreeNodeIndex,
                        MatrixBufferFloat& leftYs,
                        const int rightTreeNodeIndex,
                        MatrixBufferFloat& rightYs);

    // Data has to be passed in because ProcessData may not keep the data
    void SplitIndices(  const BufferCollection& data,
                        const MatrixBufferInt& sampleIndices,
                        MatrixBufferInt& leftSampleIndicesOut,
                        MatrixBufferInt& rightSampleIndicesOut );

private:
    // Passed in (not owned)
    const SplitCriteriaI* mSplitCriteria;
    const int mTreeDepth;

    // Created on construction
    std::vector<ActiveSplitNodeFeatureSet> mActiveSplitNodeFeatureSets;

    // Updated everytime ProcessData is called
    // Across all mActiveSplitNodeFeatureSets
    int mBestFeatureIndex;
    SPLT_CRITERIA mShouldSplit;

    MatrixBufferFloat mImpurities;
    MatrixBufferFloat mThresholds;
    MatrixBufferFloat mChildCounts;
    MatrixBufferInt mFeatureIndices;

};