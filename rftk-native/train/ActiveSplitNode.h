#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <vector>
#include <tr1/memory>

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorI.h"
#include "NodeDataCollectorI.h"
#include "BestSplitI.h"
#include "SplitCriteriaI.h"



class ActiveSplitNodeFeatureSet
{
public:
    ActiveSplitNodeFeatureSet(  const FeatureExtractorI* featureExtractor,
                                NodeDataCollectorI* nodeDataCollector,
                                const BestSplitI* bestSplitter );

    ~ActiveSplitNodeFeatureSet();

    void ProcessData(   const BufferCollection& data,
                        const Int32MatrixBuffer& sampleIndices,
                        boost::mt19937& gen );

    void WriteImpurity( int groupId,
                        int outStartIndex,
                        Float32MatrixBuffer& impuritiesOut,
                        Float32MatrixBuffer& thresholdsOut,
                        Float32MatrixBuffer& childCountsOut,
                        Int32MatrixBuffer& featureIndicesOut );

    void SplitIndices(  const int featureIndex,
                        const BufferCollection& data,
                        const Int32MatrixBuffer& sampleIndices,
                        Int32MatrixBuffer& leftSampleIndicesOut,
                        Int32MatrixBuffer& rightSampleIndicesOut );

    void WriteToTree(   int index,
                        const int treeNodeIndex,
                        Float32MatrixBuffer& floatParamsOut,
                        Int32MatrixBuffer& intParamsOut,
                        const int leftTreeNodeIndex,
                        Float32MatrixBuffer& leftYsOut,
                        const int rightTreeNodeIndex,
                        Float32MatrixBuffer& rightYsOut );

    int GetNumberFeatureCandidates() const { return mIntParams.GetM(); }

private:
    // Passed in (not owned)
    const FeatureExtractorI* mFeatureExtractor;
    const BestSplitI* mBestSplitter;

    // Passed in but owned by this class
    std::tr1::shared_ptr<NodeDataCollectorI> mNodeDataCollector;

    // Created on construction
    Int32MatrixBuffer mIntParams;
    Float32MatrixBuffer mFloatParams;

    // Updated everytime ProcessData is called
    Float32MatrixBuffer mImpurities;
    Float32MatrixBuffer mThresholds;
    Float32MatrixBuffer mChildCounts;
    Float32MatrixBuffer mLeftYs;
    Float32MatrixBuffer mRightYs;
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
                        const Int32MatrixBuffer& sampleIndices,
                        boost::mt19937& gen );

    SPLT_CRITERIA ShouldSplit() { return mShouldSplit; }

    void WriteToTree(   const int treeNodeIndex,
                        Int32MatrixBuffer& paths,
                        Float32MatrixBuffer& floatParams,
                        Int32MatrixBuffer& intParams,
                        Int32MatrixBuffer& depth,
                        const int leftTreeNodeIndex,
                        Float32MatrixBuffer& leftYs,
                        const int rightTreeNodeIndex,
                        Float32MatrixBuffer& rightYs);

    // Data has to be passed in because ProcessData may not keep the data
    void SplitIndices(  const BufferCollection& data,
                        const Int32MatrixBuffer& sampleIndices,
                        Int32MatrixBuffer& leftSampleIndicesOut,
                        Int32MatrixBuffer& rightSampleIndicesOut );

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

    Float32MatrixBuffer mImpurities;
    Float32MatrixBuffer mThresholds;
    Float32MatrixBuffer mChildCounts;
    Int32MatrixBuffer mFeatureIndices;

};