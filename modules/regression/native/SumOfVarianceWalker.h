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
template <class BT>
class SumOfVarianceWalker
{
public:
    SumOfVarianceWalker (const BufferId& sampleWeights,
                        const BufferId& ys,
                        const int ydim );
    virtual ~SumOfVarianceWalker();

    void Bind(const BufferCollectionStack& readCollection);
    void Bind(const BufferCollectionStack& readCollection, const VectorBufferTemplate<typename BT::Index>& includedSamples);
    void Reset();

    void MoveLeftToRight(typename BT::Index sampleIndex);

    typename BT::ImpurityValue Impurity();

    typename BT::Index GetYDim() const;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetLeftYs() const;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetRightYs() const;
    typename BT::SufficientStatsContinuous GetLeftChildCounts() const;
    typename BT::SufficientStatsContinuous GetRightChildCounts() const;

    typedef BT BufferTypes;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mYsBufferId;
    const int mYdim;

    VectorBufferTemplate<typename BT::ParamsContinuous> const* mSampleWeights;
    MatrixBufferTemplate<typename BT::SourceContinuous> const* mYs;

    typename BT::SufficientStatsContinuous mStartCounts;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mStartMeanVariance;
    typename BT::SufficientStatsContinuous mLeftCounts;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftMeanVariance;
    typename BT::SufficientStatsContinuous mRightCounts;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightMeanVariance;

    typename BT::SufficientStatsContinuous mStartVariance;
};


template <class BT>
SumOfVarianceWalker<BT>::SumOfVarianceWalker(const BufferId& sampleWeights,
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

template <class BT>
SumOfVarianceWalker<BT>::~SumOfVarianceWalker()
{}

template <class BT>
void SumOfVarianceWalker<BT>::Bind(const BufferCollectionStack& readCollection)
{
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::DatapointCounts> >(mSampleWeightsBufferId);

    mYs = readCollection.GetBufferPtr< MatrixBufferTemplate<typename BT::SourceContinuous> >(mYsBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mYs->GetM())

    for(int i=0; i<mSampleWeights->GetN(); i++)
    {
        const typename BT::SufficientStatsContinuous weight = mSampleWeights->Get(i);
        typename BT::SufficientStatsContinuous newCounts = mStartCounts + weight;
        for(int d=0; d<mYdim; d++)
        {
            const typename BT::SufficientStatsContinuous epsilon = typename BT::SufficientStatsContinuous(0.1);
            const typename BT::SufficientStatsContinuous zero = typename BT::SufficientStatsContinuous(0);
            const typename BT::SufficientStatsContinuous y_i = mYs->Get(i,d);
            // old unstable sufficient stats 
            // mStartMeanVariance.Incr(d, weight*y_d);
            // mStartMeanVariance.Incr(d+mYdim, weight*y_d*y_d);
            const typename BT::SufficientStatsContinuous mean = mStartMeanVariance.Get(d);
            const typename BT::SufficientStatsContinuous delta = y_i - mean;
            const typename BT::SufficientStatsContinuous r = newCounts > epsilon ? delta * weight / newCounts : zero;
            mStartMeanVariance.Incr(d,r);
            mStartMeanVariance.Incr(d+mYdim, mStartCounts*delta*r);
        }
        mStartCounts = newCounts;
    }

    mStartVariance = typename BT::SufficientStatsContinuous(0);
    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats 
        // const typename BT::SufficientStatsContinuous y = mStartMeanVariance.Get(d);
        // const typename BT::SufficientStatsContinuous ySquared = mStartMeanVariance.Get(d+mYdim);
        // mStartVariance += ySquared / mStartCounts - pow(y/mStartCounts, 2);
        mStartVariance += mStartMeanVariance.Get(d+mYdim) / mStartCounts;
    }

    Reset();
}

template <class BT>
void SumOfVarianceWalker<BT>::Bind(const BufferCollectionStack& readCollection, const VectorBufferTemplate<typename BT::Index>& includedSamples)
{
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::DatapointCounts> >(mSampleWeightsBufferId);

    mYs = readCollection.GetBufferPtr< MatrixBufferTemplate<typename BT::SourceContinuous> >(mYsBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mYs->GetM())

    for(typename BT::Index s=0; s<includedSamples.GetN(); s++)
    {
        typename BT::Index i = includedSamples.Get(s);
        const typename BT::SufficientStatsContinuous weight = mSampleWeights->Get(i);
        typename BT::SufficientStatsContinuous newCounts = mStartCounts + weight;
        for(int d=0; d<mYdim; d++)
        {
            const typename BT::SufficientStatsContinuous epsilon = typename BT::SufficientStatsContinuous(0.1);
            const typename BT::SufficientStatsContinuous zero = typename BT::SufficientStatsContinuous(0);
            const typename BT::SufficientStatsContinuous y_i = mYs->Get(i,d);
            // old unstable sufficient stats 
            // mStartMeanVariance.Incr(d, weight*y_d);
            // mStartMeanVariance.Incr(d+mYdim, weight*y_d*y_d);
            const typename BT::SufficientStatsContinuous mean = mStartMeanVariance.Get(d);
            const typename BT::SufficientStatsContinuous delta = y_i - mean;
            const typename BT::SufficientStatsContinuous r = newCounts > epsilon ? delta * weight / newCounts : zero;
            mStartMeanVariance.Incr(d,r);
            mStartMeanVariance.Incr(d+mYdim, mStartCounts*delta*r);
        }
        mStartCounts = newCounts;
    }

    mStartVariance = typename BT::SufficientStatsContinuous(0);
    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats 
        // const typename BT::SufficientStatsContinuous y = mStartMeanVariance.Get(d);
        // const typename BT::SufficientStatsContinuous ySquared = mStartMeanVariance.Get(d+mYdim);
        // mStartVariance += ySquared / mStartCounts - pow(y/mStartCounts, 2);
        mStartVariance += mStartMeanVariance.Get(d+mYdim) / mStartCounts;
    }

    Reset();
}

template <class BT>
void SumOfVarianceWalker<BT>::Reset()
{
    mLeftCounts = mStartCounts;
    mLeftMeanVariance = mStartMeanVariance;
    mRightCounts = typename BT::SufficientStatsContinuous(0);
    mRightMeanVariance.Zero();
}

template <class BT>
void SumOfVarianceWalker<BT>::MoveLeftToRight(typename BT::Index sampleIndex)
{
    const typename BT::SufficientStatsContinuous weight = mSampleWeights->Get(sampleIndex);

    const typename BT::SufficientStatsContinuous newLeftCounts = mLeftCounts - weight;
    const typename BT::SufficientStatsContinuous newRightCounts = mRightCounts + weight;

    for(int d=0; d<mYdim; d++)
    {
        const typename BT::SufficientStatsContinuous epsilon = typename BT::SufficientStatsContinuous(0.1);
        const typename BT::SufficientStatsContinuous zero = typename BT::SufficientStatsContinuous(0);

        const typename BT::SufficientStatsContinuous y_d = mYs->Get(sampleIndex,d);

        // old unstable sufficient stats 
        // mLeftMeanVariance.Incr(d, -weight*y_d);
        // mLeftMeanVariance.Incr(d+mYdim, -weight*y_d*y_d);
        // mRightMeanVariance.Incr(d, weight*y_d);
        // mRightMeanVariance.Incr(d+mYdim, weight*y_d*y_d);

        const typename BT::SufficientStatsContinuous leftMean = mLeftMeanVariance.Get(d);
        const typename BT::SufficientStatsContinuous leftDelta = y_d - leftMean;
        const typename BT::SufficientStatsContinuous leftr = newLeftCounts > epsilon ? leftDelta * -1.0 * weight / newLeftCounts : zero;
        mLeftMeanVariance.Incr(d, leftr);
        mLeftMeanVariance.Incr(d+mYdim, mLeftCounts*leftDelta*leftr);

        const typename BT::SufficientStatsContinuous rightMean = mRightMeanVariance.Get(d);
        const typename BT::SufficientStatsContinuous rightDelta = y_d - rightMean;
        const typename BT::SufficientStatsContinuous rightr = newRightCounts > epsilon ? rightDelta * weight / newRightCounts : zero;
        mRightMeanVariance.Incr(d, rightr);
        mRightMeanVariance.Incr(d+mYdim, mRightCounts*rightDelta*rightr);
    }
    mLeftCounts = newLeftCounts;
    mRightCounts = newRightCounts;
}

template <class BT>
typename BT::ImpurityValue SumOfVarianceWalker<BT>::Impurity()
{
    const typename BT::SufficientStatsContinuous countsTotal = mLeftCounts+mRightCounts;
    typename BT::SufficientStatsContinuous leftSumOfVariance = typename BT::SufficientStatsContinuous(0);
    typename BT::SufficientStatsContinuous rightSumOfVariance = typename BT::SufficientStatsContinuous(0);

    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats
        // const FloatType leftY = mLeftMeanVariance.Get(d);
        // const FloatType leftYSquared = mLeftMeanVariance.Get(d+mYdim);
        // const FloatType rightY = mRightMeanVariance.Get(d);
        // const FloatType rightYSquared = mRightMeanVariance.Get(d+mYdim);
        // leftSumOfVariance += (mLeftCounts>0.0) ? leftYSquared/mLeftCounts -  pow(leftY/mLeftCounts, 2) : 0.0;
        // rightSumOfVariance += (mRightCounts>0.0) ? rightYSquared/mRightCounts -  pow(rightY/mRightCounts, 2) : 0.0;

        const typename BT::SufficientStatsContinuous leftY2 = mLeftMeanVariance.Get(d+mYdim);
        const typename BT::SufficientStatsContinuous rightY2 = mRightMeanVariance.Get(d+mYdim);

        leftSumOfVariance += (mLeftCounts>0.0) ? leftY2/mLeftCounts : typename BT::SufficientStatsContinuous(0);
        rightSumOfVariance += (mRightCounts>0.0) ? rightY2/mRightCounts : typename BT::SufficientStatsContinuous(0);
    }

    const typename BT::ImpurityValue varianceGain = mStartVariance
                                  - ((mLeftCounts / countsTotal) * leftSumOfVariance)
                                  - ((mRightCounts / countsTotal) * rightSumOfVariance);

    return varianceGain;
}

template <class BT>
typename BT::Index SumOfVarianceWalker<BT>::GetYDim() const
{
    return mYdim*2;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> SumOfVarianceWalker<BT>::GetLeftYs() const
{
    return mLeftMeanVariance;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> SumOfVarianceWalker<BT>::GetRightYs() const
{
    return mRightMeanVariance;
}

template <class BT>
typename BT::SufficientStatsContinuous SumOfVarianceWalker<BT>::GetLeftChildCounts() const
{
    return mLeftCounts;
}

template <class BT>
typename BT::SufficientStatsContinuous SumOfVarianceWalker<BT>::GetRightChildCounts() const
{
    return mRightCounts;
}