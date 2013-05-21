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
template <class FloatType, class IntType>
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

    FloatType mStartCounts;
    VectorBufferTemplate<FloatType> mStartMeanVariance;
    FloatType mLeftCounts;
    VectorBufferTemplate<FloatType> mLeftMeanVariance;
    FloatType mRightCounts;
    VectorBufferTemplate<FloatType> mRightMeanVariance;

    FloatType mStartVariance;

    std::vector<bool> mRecomputeClassLog;
};


template <class FloatType, class IntType>
SumOfVarianceWalker<FloatType, IntType>::SumOfVarianceWalker(const BufferId& sampleWeights,
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

template <class FloatType, class IntType>
SumOfVarianceWalker<FloatType, IntType>::~SumOfVarianceWalker()
{}

template <class FloatType, class IntType>
void SumOfVarianceWalker<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection)
{
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);
    mYs = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mYsBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mYs->GetM())

    for(int i=0; i<mSampleWeights->GetN(); i++)
    {
        const FloatType weight = mSampleWeights->Get(i);
        for(int d=0; d<mYdim; d++)
        {
            const FloatType y_d = mYs->Get(i,d);
            mStartMeanVariance.Incr(d, weight*y_d);
            mStartMeanVariance.Incr(d+mYdim, weight*y_d*y_d);
        }
        mStartCounts += weight;
    }

    mStartVariance = FloatType(0);
    for(int d=0; d<mYdim; d++)
    {
        const FloatType y = mStartMeanVariance.Get(d);
        const FloatType ySquared = mStartMeanVariance.Get(d+mYdim);
        mStartVariance += ySquared / mStartCounts - pow(y/mStartCounts, 2);
    }

    Reset();
}

template <class FloatType, class IntType>
void SumOfVarianceWalker<FloatType, IntType>::Reset()
{
    mLeftCounts = mStartCounts;
    mLeftMeanVariance = mStartMeanVariance;
    mRightCounts = FloatType(0);
    mRightMeanVariance.Zero();
}

template <class FloatType, class IntType>
void SumOfVarianceWalker<FloatType, IntType>::MoveLeftToRight(IntType sampleIndex)
{
    const FloatType weight = mSampleWeights->Get(sampleIndex);

    mLeftCounts -= weight;
    mRightCounts += weight;

    for(int d=0; d<mYdim; d++)
    {
        const FloatType y_d = mYs->Get(sampleIndex,d);
        mLeftMeanVariance.Incr(d, -weight*y_d);
        mLeftMeanVariance.Incr(d+mYdim, -weight*y_d*y_d);
        mRightMeanVariance.Incr(d, weight*y_d);
        mRightMeanVariance.Incr(d+mYdim, weight*y_d*y_d);
    }
}

template <class FloatType, class IntType>
FloatType SumOfVarianceWalker<FloatType, IntType>::Impurity()
{
    const FloatType countsTotal = mLeftCounts+mRightCounts;
    FloatType leftSumOfVariance = FloatType(0);
    FloatType rightSumOfVariance = FloatType(0);

    for(int d=0; d<mYdim; d++)
    {
        const FloatType leftY = mLeftMeanVariance.Get(d);
        const FloatType leftYSquared = mLeftMeanVariance.Get(d+mYdim);
        const FloatType rightY = mRightMeanVariance.Get(d);
        const FloatType rightYSquared = mRightMeanVariance.Get(d+mYdim);

        leftSumOfVariance += (mLeftCounts>0.0) ? leftYSquared/mLeftCounts -  pow(leftY/mLeftCounts, 2) : 0.0;
        rightSumOfVariance += (mRightCounts>0.0) ? rightYSquared/mRightCounts -  pow(rightY/mRightCounts, 2) : 0.0;
    }

    const FloatType varianceGain = mStartVariance
                                  - ((mLeftCounts / countsTotal) * leftSumOfVariance)
                                  - ((mRightCounts / countsTotal) * rightSumOfVariance);

    return varianceGain;
}

template <class FloatType, class IntType>
IntType SumOfVarianceWalker<FloatType, IntType>::GetYDim() const
{
    return mYdim*2;
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> SumOfVarianceWalker<FloatType, IntType>::GetLeftYs() const
{
    return mLeftMeanVariance;
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> SumOfVarianceWalker<FloatType, IntType>::GetRightYs() const
{
    return mRightMeanVariance;
}

template <class FloatType, class IntType>
FloatType SumOfVarianceWalker<FloatType, IntType>::GetLeftChildCounts() const
{
    return mLeftCounts;
}

template <class FloatType, class IntType>
FloatType SumOfVarianceWalker<FloatType, IntType>::GetRightChildCounts() const
{
    return mRightCounts;
}