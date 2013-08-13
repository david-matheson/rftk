#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"
#include "SplitSelectorBuffers.h"
#include "ShouldSplitCriteriaI.h"
#include "FinalizerI.h"
#include "SplitBuffersI.h"

// ----------------------------------------------------------------------------
//
// SplitSelectorInfo contains the indices into the best split and allows the 
// it to be written to a tree
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class SplitSelectorInfo
{
public:
    SplitSelectorInfo(const SplitSelectorBuffers& splitStatistics,
                      const BufferCollectionStack& bufferCollectionStack,
                      const FinalizerI<BufferTypes>* finalizer,
                      const SplitBuffersI* bufferSplitter,
                      int bestFeature,
                      int bestSplitpoints,
                      int depth);

    bool ValidSplit() const;

    void WriteToTree(int nodeId, int leftNodeId, int rightNodeId,
                          VectorBufferTemplate<typename BufferTypes::DatapointCounts>& treeCounts,
                          VectorBufferTemplate<typename BufferTypes::Index>& depths,
                          MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& treeFloatFeatureParams,
                          MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& treeIntFeatureParams,
                          MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& treeFloatEstimatorParams ) const;

    void SplitBuffers(BufferCollection& leftBuffers, 
                      BufferCollection& rightBuffers,
                      typename BufferTypes::DatapointCounts& leftSize, 
                      typename BufferTypes::DatapointCounts& rightSize) const;

private:
    const SplitSelectorBuffers& mSplitSelectorBuffers;
    const BufferCollectionStack& mReadCollection;
    const FinalizerI<BufferTypes>* mFinalizer;
    const SplitBuffersI* mBufferSplitter;
    const int mBestFeature;
    const int mBestSplitpoint;
    const int mDepth;
};


// TODO: Move to a better location
enum
{
    SPLIT_SELECTOR_NO_SPLIT = -1,
    LEFT_CHILD_INDEX = 0,
    RIGHT_CHILD_INDEX = 1,
    SPLITPOINT_INDEX = 0
};

template <class BufferTypes>
SplitSelectorInfo<BufferTypes>::SplitSelectorInfo(const SplitSelectorBuffers& splitStatistics,
                                                        const BufferCollectionStack& bufferCollectionStack,
                                                        const FinalizerI<BufferTypes>* finalizer,
                                                        const SplitBuffersI* bufferSplitter,
                                                        int bestFeature,
                                                        int bestSplitpoints,
                                                        int depth)
: mSplitSelectorBuffers(splitStatistics)
, mReadCollection(bufferCollectionStack)
, mFinalizer(finalizer)
, mBufferSplitter(bufferSplitter)
, mBestFeature(bestFeature)
, mBestSplitpoint(bestSplitpoints)
, mDepth(depth)
{}

template <class BufferTypes>
bool SplitSelectorInfo<BufferTypes>::ValidSplit() const
{
    return (mBestFeature != SPLIT_SELECTOR_NO_SPLIT && mBestSplitpoint != SPLIT_SELECTOR_NO_SPLIT);
}

template <class BufferTypes>
void SplitSelectorInfo<BufferTypes>::WriteToTree(int nodeId, int leftNodeId, int rightNodeId,
                                                  VectorBufferTemplate<typename BufferTypes::DatapointCounts>& treeCounts,
                                                  VectorBufferTemplate<typename BufferTypes::Index>& depths,
                                                  MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& treeFloatFeatureParams,
                                                  MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& treeIntFeatureParams,
                                                  MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& treeFloatEstimatorParams  ) const
{
    ASSERT(ValidSplit())

    const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitpoints
          = mReadCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(mSplitSelectorBuffers.mSplitpointsBufferId);

    const Tensor3BufferTemplate<typename BufferTypes::DatapointCounts>& childCounts
           = mReadCollection.GetBuffer< Tensor3BufferTemplate<typename BufferTypes::DatapointCounts> >(mSplitSelectorBuffers.mChildCountsBufferId);

    const Tensor3BufferTemplate<typename BufferTypes::SufficientStatsContinuous>& leftEstimatorParams
           = mReadCollection.GetBuffer< Tensor3BufferTemplate<typename BufferTypes::SufficientStatsContinuous> >(mSplitSelectorBuffers.mLeftEstimatorParamsBufferId);

    const Tensor3BufferTemplate<typename BufferTypes::SufficientStatsContinuous>& rightEstimatorParams
           = mReadCollection.GetBuffer< Tensor3BufferTemplate<typename BufferTypes::SufficientStatsContinuous> >(mSplitSelectorBuffers.mRightEstimatorParamsBufferId);

    const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams
          = mReadCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mSplitSelectorBuffers.mFloatParamsBufferId);

    const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams
          = mReadCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mSplitSelectorBuffers.mIntParamsBufferId);

    depths.Set(nodeId, mDepth);
    depths.Set(leftNodeId, mDepth+1);
    depths.Set(rightNodeId, mDepth+1);

    treeIntFeatureParams.SetRow(nodeId, intParams.SliceRowAsVector(mBestFeature));
    treeFloatFeatureParams.SetRow(nodeId, floatParams.SliceRowAsVector(mBestFeature));

    const typename BufferTypes::FeatureValue bestSplitpointValue = splitpoints.Get(mBestFeature, mBestSplitpoint);
    treeFloatFeatureParams.Set(nodeId, SPLITPOINT_INDEX, bestSplitpointValue); 

    const typename BufferTypes::DatapointCounts leftCounts = childCounts.Get(mBestFeature, mBestSplitpoint, LEFT_CHILD_INDEX); 
    treeCounts.Set(leftNodeId, leftCounts);
    VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous> leftEstimatorParamsVector 
            = leftEstimatorParams.SliceRow(mBestFeature, mBestSplitpoint);
    treeFloatEstimatorParams.SetRow(leftNodeId, mFinalizer->Finalize(leftCounts, leftEstimatorParamsVector));

    const typename BufferTypes::DatapointCounts rightCounts = childCounts.Get(mBestFeature, mBestSplitpoint, RIGHT_CHILD_INDEX); 
    treeCounts.Set(rightNodeId, rightCounts);
    VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous> rightEstimatorParamsVector 
            = rightEstimatorParams.SliceRow(mBestFeature, mBestSplitpoint);
    treeFloatEstimatorParams.SetRow(rightNodeId, mFinalizer->Finalize(rightCounts, rightEstimatorParamsVector));
}

template <class BufferTypes>
void SplitSelectorInfo<BufferTypes>::SplitBuffers(BufferCollection& leftBuffers, BufferCollection& rightBuffers,
                                                          typename BufferTypes::DatapointCounts& leftSize, typename BufferTypes::DatapointCounts& rightSize) const
{
    ASSERT(ValidSplit())

    const Tensor3BufferTemplate<typename BufferTypes::DatapointCounts>& childCounts
           = mReadCollection.GetBuffer< Tensor3BufferTemplate<typename BufferTypes::DatapointCounts> >(mSplitSelectorBuffers.mChildCountsBufferId);
    leftSize = childCounts.Get(mBestFeature, mBestSplitpoint, LEFT_CHILD_INDEX);
    rightSize = childCounts.Get(mBestFeature, mBestSplitpoint, RIGHT_CHILD_INDEX); 

    if( mBufferSplitter == NULL)
    {
        printf("Error: trying to split a node's buffers without a SplitBuffersI.\n");
        printf("Did you forget to pass SplitBuffersIndices() to SplitSelector or WaitForBestSplitSelector?\n");
    }
    else
    {
        mBufferSplitter->SplitBuffers(mSplitSelectorBuffers,
                                mBestFeature,
                                mBestSplitpoint,
                                mReadCollection,
                                leftBuffers,
                                rightBuffers);
    }
}