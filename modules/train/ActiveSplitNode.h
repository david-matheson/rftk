#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <vector>
#include <tr1/memory>

#include "buffers/MatrixBuffer.h"
#include "buffers/BufferCollection.h"
#include "feature_extractors/FeatureExtractorI.h"
#include "best_split/BestSplitI.h"

#include "NodeDataCollectorI.h"
#include "SplitCriteriaI.h"



class ActiveSplitNodeFeatureSet
{
public:
    ActiveSplitNodeFeatureSet(  const FeatureExtractorI* featureExtractor,
                                NodeDataCollectorI* nodeDataCollector,
                                const BestSplitI* bestSplitter,
                                const int evalSplitPeriod );

    ~ActiveSplitNodeFeatureSet();

    ActiveSplitNodeFeatureSet( const ActiveSplitNodeFeatureSet& other );
    ActiveSplitNodeFeatureSet& operator=( const ActiveSplitNodeFeatureSet& rhs );

    void ProcessData(   const BufferCollection& data,
                        const Int32VectorBuffer& sampleIndices,
                        boost::mt19937& gen,
                        const int minSamples );

    void WriteImpurity( int groupId,
                        int outStartIndex,
                        Float32VectorBuffer& impuritiesOut,
                        Float32VectorBuffer& thresholdsOut,
                        Float32MatrixBuffer& childCountsOut,
                        Int32MatrixBuffer& featureIndicesOut );

    void SplitIndices(  const int featureIndex,
                        const BufferCollection& data,
                        const Int32VectorBuffer& sampleIndices,
                        Int32VectorBuffer& leftSampleIndicesOut,
                        Int32VectorBuffer& rightSampleIndicesOut );

    void WriteToTree(   int index,
                        const int treeNodeIndex,
                        const int leftTreeNodeIndex,
                        const int rightTreeNodeIndex,
                        Float32MatrixBuffer& floatParamsOut,
                        Int32MatrixBuffer& intParamsOut,
                        Float32VectorBuffer& countsOut,
                        Float32MatrixBuffer& ysOut );

    int GetNumberFeatureCandidates() const { return mIntParams.GetM(); }

private:
    // Passed in (not owned)
    const FeatureExtractorI* mFeatureExtractor;
    const BestSplitI* mBestSplitter;
    int mNumberOfFeatures;

    // Passed in but owned by this class
    std::tr1::shared_ptr<NodeDataCollectorI> mNodeDataCollector;

    int mEvalSplitPeriod;
    int mNumberSamplesToEvalSplit;

    // Created on construction
    Int32MatrixBuffer mIntParams;
    Float32MatrixBuffer mFloatParams;

    // Updated everytime ProcessData is called
    Float32VectorBuffer mImpurities;
    Float32VectorBuffer mThresholds;
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
                    const int treeDepth,
                    const int evalSplitPeriod );

    virtual ~ActiveSplitNode();




    void ProcessData(   const BufferCollection& data,
                        const Int32VectorBuffer& sampleIndices,
                        boost::mt19937& gen );

    SPLT_CRITERIA ShouldSplit() { return mShouldSplit; }

    void WriteToTree(   const int treeNodeIndex,
                        const int leftTreeNodeIndex,
                        const int rightTreeNodeIndex,
                        Int32MatrixBuffer& paths,
                        Float32MatrixBuffer& floatParams,
                        Int32MatrixBuffer& intParams,
                        Int32VectorBuffer& depth,
                        Float32VectorBuffer& counts,
                        Float32MatrixBuffer& ys);

    // Data has to be passed in because ProcessData may not keep the data
    void SplitIndices(  const BufferCollection& data,
                        const Int32VectorBuffer& sampleIndices,
                        Int32VectorBuffer& leftSampleIndicesOut,
                        Int32VectorBuffer& rightSampleIndicesOut );

private:
    ActiveSplitNode( const ActiveSplitNode& other );
    ActiveSplitNode& operator=( const ActiveSplitNode& rhs );

    // Passed in (not owned)
    const SplitCriteriaI* mSplitCriteria;
    const int mTreeDepth;

    // Created on construction
    std::vector<ActiveSplitNodeFeatureSet> mActiveSplitNodeFeatureSets;

    // Updated everytime ProcessData is called
    // Across all mActiveSplitNodeFeatureSets
    int mBestFeatureIndex;
    SPLT_CRITERIA mShouldSplit;

    Float32VectorBuffer mImpurities;
    Float32VectorBuffer mThresholds;
    Float32MatrixBuffer mChildCounts;
    Int32MatrixBuffer mFeatureIndices;

};