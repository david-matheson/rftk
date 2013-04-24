#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"
#include "SplitSelectorBuffers.h"
#include "ShouldSplitCriteriaI.h"
#include "FinalizerI.h"

// ----------------------------------------------------------------------------
//
// SplitSelectorInfo contains the indices into the best split and allows the 
// it to be written to a tree
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class SplitSelectorInfo
{
public:
    SplitSelectorInfo(const SplitSelectorBuffers& splitStatistics,
                      const BufferCollectionStack& bufferCollectionStack,
                      const FinalizerI<FloatType>* finalizer,
                      int bestFeature,
                      int bestSplitpoints);

    bool ValidSplit() const;

    void WriteToTree(int nodeId, int leftNodeId, int rightNodeId,
                          VectorBufferTemplate<FloatType>& treeCounts,
                          MatrixBufferTemplate<FloatType>& treeFloatFeatureParams,
                          MatrixBufferTemplate<IntType>& treeIntFeatureParams,
                          MatrixBufferTemplate<FloatType>& treeFloatEstimatorParams ) const;

private:
    const SplitSelectorBuffers& mSplitSelectorBuffers;
    const BufferCollectionStack& mReadCollection;
    const FinalizerI<FloatType>* mFinalizer;
    const int mBestFeature;
    const int mBestSplitpoint;
};


// TODO: Move to a better location
enum
{
    SPLIT_SELECTOR_NO_SPLIT = -1,
    LEFT_CHILD_INDEX = 0,
    RIGHT_CHILD_INDEX = 1,
    SPLITPOINT_INDEX = 0
};

template <class FloatType, class IntType>
SplitSelectorInfo<FloatType, IntType>::SplitSelectorInfo(const SplitSelectorBuffers& splitStatistics,
                                                        const BufferCollectionStack& bufferCollectionStack,
                                                        const FinalizerI<FloatType>* finalizer,
                                                        int bestFeature,
                                                        int bestSplitpoints)
: mSplitSelectorBuffers(splitStatistics)
, mReadCollection(bufferCollectionStack)
, mFinalizer(finalizer)
, mBestFeature(bestFeature)
, mBestSplitpoint(bestSplitpoints)
{}

template <class FloatType, class IntType>
bool SplitSelectorInfo<FloatType, IntType>::ValidSplit() const
{
    return (mBestFeature != SPLIT_SELECTOR_NO_SPLIT && mBestSplitpoint != SPLIT_SELECTOR_NO_SPLIT);
}

template <class FloatType, class IntType>
void SplitSelectorInfo<FloatType, IntType>::WriteToTree(int nodeId, int leftNodeId, int rightNodeId,
                                                          VectorBufferTemplate<FloatType>& treeCounts,
                                                          MatrixBufferTemplate<FloatType>& treeFloatFeatureParams,
                                                          MatrixBufferTemplate<IntType>& treeIntFeatureParams,
                                                          MatrixBufferTemplate<FloatType>& treeFloatEstimatorParams ) const
{
    ASSERT(ValidSplit())

    const MatrixBufferTemplate<FloatType>& splitpoints
          = mReadCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(mSplitSelectorBuffers.mSplitpointsBufferId);

    const Tensor3BufferTemplate<FloatType>& childCounts
           = mReadCollection.GetBuffer< Tensor3BufferTemplate<FloatType> >(mSplitSelectorBuffers.mChildCountsBufferId);

    const Tensor3BufferTemplate<FloatType>& leftEstimatorParams
           = mReadCollection.GetBuffer< Tensor3BufferTemplate<FloatType> >(mSplitSelectorBuffers.mLeftEstimatorParamsBufferId);

    const Tensor3BufferTemplate<FloatType>& rightEstimatorParams
           = mReadCollection.GetBuffer< Tensor3BufferTemplate<FloatType> >(mSplitSelectorBuffers.mRightEstimatorParamsBufferId);

    const MatrixBufferTemplate<FloatType>& floatParams
          = mReadCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(mSplitSelectorBuffers.mFloatParamsBufferId);

    const MatrixBufferTemplate<IntType>& intParams
          = mReadCollection.GetBuffer< MatrixBufferTemplate<IntType> >(mSplitSelectorBuffers.mIntParamsBufferId);

    treeIntFeatureParams.SetRow(nodeId, intParams.SliceRowAsVector(mBestFeature));
    treeFloatFeatureParams.SetRow(nodeId, floatParams.SliceRowAsVector(mBestFeature));

    const FloatType bestSplitSplitpoints = splitpoints.Get(mBestFeature, mBestSplitpoint);
    treeFloatFeatureParams.Set(nodeId, SPLITPOINT_INDEX, bestSplitSplitpoints); 

    const FloatType leftCounts = childCounts.Get(mBestFeature, mBestSplitpoint, LEFT_CHILD_INDEX); 
    treeCounts.Set(leftNodeId, leftCounts);
    VectorBufferTemplate<FloatType> leftEstimatorParamsVector = leftEstimatorParams.SliceRow(mBestFeature, mBestSplitpoint);
    mFinalizer->Finalize(leftCounts, leftEstimatorParamsVector);
    treeFloatEstimatorParams.SetRow(leftNodeId, leftEstimatorParamsVector);

    const FloatType rightCounts = childCounts.Get(mBestFeature, mBestSplitpoint, RIGHT_CHILD_INDEX); 
    treeCounts.Set(rightNodeId, rightCounts);
    VectorBufferTemplate<FloatType> rightEstimatorParamsVector = rightEstimatorParams.SliceRow(mBestFeature, mBestSplitpoint);
    mFinalizer->Finalize(rightCounts, rightEstimatorParamsVector);
    treeFloatEstimatorParams.SetRow(rightNodeId, rightEstimatorParamsVector);
}