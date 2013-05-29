#pragma once

#include <vector>
#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "AssignStreamStep.h"

// ----------------------------------------------------------------------------
//
// Compute the sum of the variance for each component
// This class is called from BestSplitpointsWalkingSortedStep where
// MoveLeftToRight is called for the sorted feature values
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType, class InternalFloatType>
class SumOfVarianceTwoStreamWalker
{
public:
    SumOfVarianceTwoStreamWalker (const BufferId& sampleWeights,
                        const BufferId& streamType,
                        const BufferId& ys,
                        const int ydim );
    virtual ~SumOfVarianceTwoStreamWalker();

    void Bind(const BufferCollectionStack& readCollection);
    void Reset();

    void MoveLeftToRight(IntType sampleIndex);

    FloatType Impurity();

    IntType GetYDim() const;
    VectorBufferTemplate<FloatType> GetLeftEstimationYs() const;
    VectorBufferTemplate<FloatType> GetLeftStructureYs() const;
    VectorBufferTemplate<FloatType> GetRightEstimationYs() const;
    VectorBufferTemplate<FloatType> GetRightStructureYs() const;
    FloatType GetLeftEstimationChildCounts() const;
    FloatType GetLeftStructureChildCounts() const;
    FloatType GetRightEstimationChildCounts() const;
    FloatType GetRightStructureChildCounts() const;

    typedef FloatType Float;
    typedef IntType Int;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mStreamTypeBufferId;
    const BufferId mYsBufferId;
    const int mYdim;

    VectorBufferTemplate<FloatType> const* mSampleWeights;
    VectorBufferTemplate<IntType> const* mStreamType;
    MatrixBufferTemplate<FloatType> const* mYs;

    InternalFloatType mStartEstimationCounts;
    InternalFloatType mStartStructureCounts;
    VectorBufferTemplate<InternalFloatType> mStartEstimationMeanVariance;
    VectorBufferTemplate<InternalFloatType> mStartStructureMeanVariance;
    InternalFloatType mLeftEstimationCounts;
    InternalFloatType mLeftStructureCounts;
    VectorBufferTemplate<InternalFloatType> mLeftEstimationMeanVariance;
    VectorBufferTemplate<InternalFloatType> mLeftStructureMeanVariance;
    InternalFloatType mRightEstimationCounts;
    InternalFloatType mRightStructureCounts;
    VectorBufferTemplate<InternalFloatType> mRightEstimationMeanVariance;
    VectorBufferTemplate<InternalFloatType> mRightStructureMeanVariance;

    InternalFloatType mStartVariance;
};


template <class FloatType, class IntType, class InternalFloatType>
SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::SumOfVarianceTwoStreamWalker(const BufferId& sampleWeights,
                                                          const BufferId& streamType,
                                                          const BufferId& ys,
                                                          const int ydim )
: mSampleWeightsBufferId(sampleWeights)
, mStreamTypeBufferId(streamType)
, mYsBufferId(ys)
, mYdim(ydim)
, mSampleWeights(NULL)
, mStreamType(NULL)
, mYs(NULL)
, mStartEstimationCounts(0)
, mStartStructureCounts(0)
, mStartEstimationMeanVariance(ydim*2)
, mStartStructureMeanVariance(ydim*2)
, mLeftEstimationCounts(0)
, mLeftStructureCounts(0)
, mLeftEstimationMeanVariance(ydim*2)
, mLeftStructureMeanVariance(ydim*2)
, mRightEstimationCounts(0)
, mRightStructureCounts(0)
, mRightEstimationMeanVariance(ydim*2)
, mRightStructureMeanVariance(ydim*2)
, mStartVariance(0)
{}

template <class FloatType, class IntType, class InternalFloatType>
SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::~SumOfVarianceTwoStreamWalker()
{}

template <class FloatType, class IntType, class InternalFloatType>
void SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::Bind(const BufferCollectionStack& readCollection)
{
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);
    mYs = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mYsBufferId);
    mStreamType = readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mStreamTypeBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mYs->GetM())

    for(int i=0; i<mSampleWeights->GetN(); i++)
    {
        const InternalFloatType weight = mSampleWeights->Get(i);
        const IntType streamType = mStreamType->Get(i);

        VectorBufferTemplate<InternalFloatType>& meanVariance = (streamType == STREAM_ESTIMATION) ?
                    mStartEstimationMeanVariance : mStartStructureMeanVariance;

        InternalFloatType& counts = (streamType == STREAM_ESTIMATION) ? mStartEstimationCounts : mStartStructureCounts;
        InternalFloatType newCounts = counts + weight;

        for(int d=0; d<mYdim; d++)
        {
            const InternalFloatType y_i = mYs->Get(i,d);
            // old unstable sufficient stats 
            // meanVariance.Incr(d, weight*y_d);
            // meanVariance.Incr(d+mYdim, weight*y_d*y_d);
            const InternalFloatType mean = meanVariance.Get(d);
            const InternalFloatType delta = y_i - mean;
            const InternalFloatType r = delta * weight / newCounts;
            meanVariance.Incr(d,r);
            meanVariance.Incr(d+mYdim, counts*delta*r);
        }
        counts = newCounts;
    }

    mStartVariance = InternalFloatType(0);
    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats 
        // const FloatType y = mStartStructureMeanVariance.Get(d);
        // const FloatType ySquared = mStartStructureMeanVariance.Get(d+mYdim);
        // mStartVariance += ySquared / mStartStructureCounts - pow(y/mStartStructureCounts, 2);
        mStartVariance += mStartStructureMeanVariance.Get(d+mYdim) / mStartStructureCounts;
    }

    Reset();
}

template <class FloatType, class IntType, class InternalFloatType>
void SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::Reset()
{
    mLeftEstimationCounts = mStartEstimationCounts;
    mLeftStructureCounts = mStartStructureCounts;
    mLeftEstimationMeanVariance = mStartEstimationMeanVariance;
    mLeftStructureMeanVariance = mStartStructureMeanVariance;
    mRightEstimationCounts = InternalFloatType(0);
    mRightStructureCounts = InternalFloatType(0);
    mRightEstimationMeanVariance.Zero();
    mRightStructureMeanVariance.Zero();
}

template <class FloatType, class IntType, class InternalFloatType>
void SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::MoveLeftToRight(IntType sampleIndex)
{
    const InternalFloatType weight = mSampleWeights->Get(sampleIndex);
    const IntType streamType = mStreamType->Get(sampleIndex);

    VectorBufferTemplate<InternalFloatType>& leftMeanVariance = (streamType == STREAM_ESTIMATION) ?
                    mLeftEstimationMeanVariance : mLeftStructureMeanVariance;

    VectorBufferTemplate<InternalFloatType>& rightMeanVariance = (streamType == STREAM_ESTIMATION) ?
                    mRightEstimationMeanVariance : mRightStructureMeanVariance;

    InternalFloatType& leftCounts = (streamType == STREAM_ESTIMATION) ?
                    mLeftEstimationCounts : mLeftStructureCounts;

    InternalFloatType& rightCounts = (streamType == STREAM_ESTIMATION) ?
                    mRightEstimationCounts : mRightStructureCounts;

    const InternalFloatType newLeftCounts = leftCounts - weight;
    const InternalFloatType newRightCounts = rightCounts + weight;

    
    for(int d=0; d<mYdim; d++)
    {
        const InternalFloatType y_d = mYs->Get(sampleIndex,d);

        // old unstable sufficient stats 
        // leftMeanVariance.Incr(d, -weight*y_d);
        // leftMeanVariance.Incr(d+mYdim, -weight*y_d*y_d);
        // rightMeanVariance.Incr(d, weight*y_d);
        // rightMeanVariance.Incr(d+mYdim, weight*y_d*y_d);

        const InternalFloatType leftMean = leftMeanVariance.Get(d);
        const InternalFloatType leftDelta = y_d - leftMean;
        const InternalFloatType leftr = leftDelta * -1.0 * weight / newLeftCounts;
        leftMeanVariance.Incr(d, leftr);
        leftMeanVariance.Incr(d+mYdim, leftCounts*leftDelta*leftr);

        const InternalFloatType rightMean = rightMeanVariance.Get(d);
        const InternalFloatType rightDelta = y_d - rightMean;
        const InternalFloatType rightr = rightDelta * weight / newRightCounts;
        rightMeanVariance.Incr(d, rightr);
        rightMeanVariance.Incr(d+mYdim, rightCounts*rightDelta*rightr);
    }
    leftCounts = newLeftCounts;
    rightCounts = newRightCounts;
}

template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::Impurity()
{
    const InternalFloatType countsTotal = mLeftStructureCounts+mRightStructureCounts;
    InternalFloatType leftSumOfVariance = FloatType(0);
    InternalFloatType rightSumOfVariance = FloatType(0);

    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats 
        // const FloatType leftY = mLeftStructureMeanVariance.Get(d);
        // const FloatType leftYSquared = mLeftStructureMeanVariance.Get(d+mYdim);
        // const FloatType rightY = mRightStructureMeanVariance.Get(d);
        // const FloatType rightYSquared = mRightStructureMeanVariance.Get(d+mYdim);

        // leftSumOfVariance += (mLeftStructureCounts>0.0) ? leftYSquared/mLeftStructureCounts -  pow(leftY/mLeftStructureCounts, 2) : 0.0;
        // rightSumOfVariance += (mRightStructureCounts>0.0) ? rightYSquared/mRightStructureCounts -  pow(rightY/mRightStructureCounts, 2) : 0.0;

        const InternalFloatType leftY2 = mLeftStructureMeanVariance.Get(d+mYdim);
        const InternalFloatType rightY2 = mRightStructureMeanVariance.Get(d+mYdim);

        leftSumOfVariance += (mLeftStructureCounts>0.0) ? leftY2/mLeftStructureCounts : InternalFloatType(0);
        rightSumOfVariance += (mRightStructureCounts>0.0) ? rightY2/mRightStructureCounts : InternalFloatType(0);
    }

    const InternalFloatType varianceGain = mStartVariance
                                  - ((mLeftStructureCounts / countsTotal) * leftSumOfVariance)
                                  - ((mRightStructureCounts / countsTotal) * rightSumOfVariance);

    return static_cast<FloatType>(varianceGain);
}

template <class FloatType, class IntType, class InternalFloatType>
IntType SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetYDim() const
{
    return mYdim*2;
}

template <class FloatType, class IntType, class InternalFloatType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetLeftEstimationYs() const
{
    return ConvertVectorBufferTemplate<InternalFloatType, FloatType>(mLeftEstimationMeanVariance);
}

template <class FloatType, class IntType, class InternalFloatType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetLeftStructureYs() const
{
    return ConvertVectorBufferTemplate<InternalFloatType, FloatType>(mLeftStructureMeanVariance);
}

template <class FloatType, class IntType, class InternalFloatType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetRightEstimationYs() const
{
    return ConvertVectorBufferTemplate<InternalFloatType, FloatType>(mRightEstimationMeanVariance);
}

template <class FloatType, class IntType, class InternalFloatType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetRightStructureYs() const
{
    return ConvertVectorBufferTemplate<InternalFloatType, FloatType>(mRightStructureMeanVariance);
}


template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetLeftEstimationChildCounts() const
{
    return mLeftEstimationCounts;
}

template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetLeftStructureChildCounts() const
{
    return mLeftStructureCounts;
}

template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetRightEstimationChildCounts() const
{
    return mRightEstimationCounts;
}

template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType, InternalFloatType>::GetRightStructureChildCounts() const
{
    return mRightStructureCounts;
}