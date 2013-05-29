#pragma once

#include <vector>
#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Compute the sum of the variance for each component
// This class is called from BestSplitpointsWalkingSortedStep where
// MoveLeftToRight is called for the sorted feature values
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType, class InternalFloatType>
class SumOfVarianceWalker
{
public:
    SumOfVarianceWalker (const BufferId& sampleWeights,
                        const BufferId& ys,
                        const int ydim );
    virtual ~SumOfVarianceWalker();

    void Bind(const BufferCollectionStack& readCollection);
    void Reset();

    void MoveLeftToRight(IntType sampleIndex);

    FloatType Impurity();

    IntType GetYDim() const;
    VectorBufferTemplate<FloatType> GetLeftYs() const;
    VectorBufferTemplate<FloatType> GetRightYs() const;
    FloatType GetLeftChildCounts() const;
    FloatType GetRightChildCounts() const;

    typedef FloatType Float;
    typedef IntType Int;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mYsBufferId;
    const int mYdim;

    VectorBufferTemplate<FloatType> const* mSampleWeights;
    MatrixBufferTemplate<FloatType> const* mYs;

    InternalFloatType mStartCounts;
    VectorBufferTemplate<InternalFloatType> mStartMeanVariance;
    InternalFloatType mLeftCounts;
    VectorBufferTemplate<InternalFloatType> mLeftMeanVariance;
    InternalFloatType mRightCounts;
    VectorBufferTemplate<InternalFloatType> mRightMeanVariance;

    InternalFloatType mStartVariance;
};


template <class FloatType, class IntType, class InternalFloatType>
SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::SumOfVarianceWalker(const BufferId& sampleWeights,
                                                          const BufferId& ys,
                                                          const int ydim )
: mSampleWeightsBufferId(sampleWeights)
, mYsBufferId(ys)
, mYdim(ydim)
, mSampleWeights(NULL)
, mYs(NULL)
, mStartCounts(0)
, mStartMeanVariance(ydim*2)
, mLeftCounts(0)
, mLeftMeanVariance(ydim*2)
, mRightCounts(0)
, mRightMeanVariance(ydim*2)
, mStartVariance(0)
{}

template <class FloatType, class IntType, class InternalFloatType>
SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::~SumOfVarianceWalker()
{}

template <class FloatType, class IntType, class InternalFloatType>
void SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::Bind(const BufferCollectionStack& readCollection)
{
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);

    mYs = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mYsBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mYs->GetM())

    for(int i=0; i<mSampleWeights->GetN(); i++)
    {
        const InternalFloatType weight = mSampleWeights->Get(i);
        InternalFloatType newCounts = mStartCounts + weight;
        for(int d=0; d<mYdim; d++)
        {
            const InternalFloatType y_i = mYs->Get(i,d);
            // old unstable sufficient stats 
            // mStartMeanVariance.Incr(d, weight*y_d);
            // mStartMeanVariance.Incr(d+mYdim, weight*y_d*y_d);
            const InternalFloatType mean = mStartMeanVariance.Get(d);
            const InternalFloatType delta = y_i - mean;
            const InternalFloatType r = delta * weight / newCounts;
            mStartMeanVariance.Incr(d,r);
            mStartMeanVariance.Incr(d+mYdim, mStartCounts*delta*r);
        }
        mStartCounts = newCounts;
    }

    mStartVariance = InternalFloatType(0);
    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats 
        // const InternalFloatType y = mStartMeanVariance.Get(d);
        // const InternalFloatType ySquared = mStartMeanVariance.Get(d+mYdim);
        // mStartVariance += ySquared / mStartCounts - pow(y/mStartCounts, 2);
        mStartVariance += mStartMeanVariance.Get(d+mYdim) / mStartCounts;
    }

    Reset();
}

template <class FloatType, class IntType, class InternalFloatType>
void SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::Reset()
{
    mLeftCounts = mStartCounts;
    mLeftMeanVariance = mStartMeanVariance;
    mRightCounts = InternalFloatType(0);
    mRightMeanVariance.Zero();
}

template <class FloatType, class IntType, class InternalFloatType>
void SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::MoveLeftToRight(IntType sampleIndex)
{
    const InternalFloatType weight = mSampleWeights->Get(sampleIndex);

    const InternalFloatType newLeftCounts = mLeftCounts - weight;
    const InternalFloatType newRightCounts = mRightCounts + weight;

    for(int d=0; d<mYdim; d++)
    {
        const InternalFloatType y_d = mYs->Get(sampleIndex,d);

        // old unstable sufficient stats 
        // mLeftMeanVariance.Incr(d, -weight*y_d);
        // mLeftMeanVariance.Incr(d+mYdim, -weight*y_d*y_d);
        // mRightMeanVariance.Incr(d, weight*y_d);
        // mRightMeanVariance.Incr(d+mYdim, weight*y_d*y_d);

        const InternalFloatType leftMean = mLeftMeanVariance.Get(d);
        const InternalFloatType leftDelta = y_d - leftMean;
        const InternalFloatType leftr = leftDelta * -1.0 * weight / newLeftCounts;
        mLeftMeanVariance.Incr(d, leftr);
        mLeftMeanVariance.Incr(d+mYdim, mLeftCounts*leftDelta*leftr);

        const InternalFloatType rightMean = mRightMeanVariance.Get(d);
        const InternalFloatType rightDelta = y_d - rightMean;
        const InternalFloatType rightr = rightDelta * weight / newRightCounts;
        mRightMeanVariance.Incr(d, rightr);
        mRightMeanVariance.Incr(d+mYdim, mRightCounts*rightDelta*rightr);
    }
    mLeftCounts = newLeftCounts;
    mRightCounts = newRightCounts;
}

template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::Impurity()
{
    const InternalFloatType countsTotal = mLeftCounts+mRightCounts;
    InternalFloatType leftSumOfVariance = FloatType(0);
    InternalFloatType rightSumOfVariance = FloatType(0);

    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats
        // const FloatType leftY = mLeftMeanVariance.Get(d);
        // const FloatType leftYSquared = mLeftMeanVariance.Get(d+mYdim);
        // const FloatType rightY = mRightMeanVariance.Get(d);
        // const FloatType rightYSquared = mRightMeanVariance.Get(d+mYdim);
        // leftSumOfVariance += (mLeftCounts>0.0) ? leftYSquared/mLeftCounts -  pow(leftY/mLeftCounts, 2) : 0.0;
        // rightSumOfVariance += (mRightCounts>0.0) ? rightYSquared/mRightCounts -  pow(rightY/mRightCounts, 2) : 0.0;

        const InternalFloatType leftY2 = mLeftMeanVariance.Get(d+mYdim);
        const InternalFloatType rightY2 = mRightMeanVariance.Get(d+mYdim);

        leftSumOfVariance += (mLeftCounts>0.0) ? leftY2/mLeftCounts : FloatType(0);
        rightSumOfVariance += (mRightCounts>0.0) ? rightY2/mRightCounts : FloatType(0);
    }

    const InternalFloatType varianceGain = mStartVariance
                                  - ((mLeftCounts / countsTotal) * leftSumOfVariance)
                                  - ((mRightCounts / countsTotal) * rightSumOfVariance);

    return static_cast<FloatType>(varianceGain);
}

template <class FloatType, class IntType, class InternalFloatType>
IntType SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::GetYDim() const
{
    return mYdim*2;
}

template <class FloatType, class IntType, class InternalFloatType>
VectorBufferTemplate<FloatType> SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::GetLeftYs() const
{
    return ConvertVectorBufferTemplate<InternalFloatType, FloatType>(mLeftMeanVariance);
}

template <class FloatType, class IntType, class InternalFloatType>
VectorBufferTemplate<FloatType> SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::GetRightYs() const
{
    return ConvertVectorBufferTemplate<InternalFloatType, FloatType>(mRightMeanVariance);
}

template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::GetLeftChildCounts() const
{
    return mLeftCounts;
}

template <class FloatType, class IntType, class InternalFloatType>
FloatType SumOfVarianceWalker<FloatType, IntType, InternalFloatType>::GetRightChildCounts() const
{
    return mRightCounts;
}