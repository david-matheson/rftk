#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class BindedMeanVarianceStatsUpdater
{
public:
    BindedMeanVarianceStatsUpdater(VectorBufferTemplate<FloatType> const* sampleWeights,
                            MatrixBufferTemplate<FloatType> const* ys);

    void UpdateStats(FloatType& counts, Tensor3BufferTemplate<FloatType>& stats,
                int feature, int threshold, int sampleIndex) const;


private:
    VectorBufferTemplate<FloatType> const* mSampleWeights;
    MatrixBufferTemplate<FloatType> const* mYs;
};

template <class FloatType, class IntType>
BindedMeanVarianceStatsUpdater<FloatType, IntType>::BindedMeanVarianceStatsUpdater(VectorBufferTemplate<FloatType> const* sampleWeights,
                                                                                  MatrixBufferTemplate<FloatType> const* ys)
: mSampleWeights(sampleWeights)
, mYs(ys)
{}

template <class FloatType, class IntType>
void BindedMeanVarianceStatsUpdater<FloatType, IntType>::UpdateStats(FloatType& counts, Tensor3BufferTemplate<FloatType>& stats,
                                                          int feature, int threshold, int sampleIndex) const
{
    const FloatType weight = mSampleWeights->Get(sampleIndex);
    const VectorBufferTemplate<FloatType> y = mYs->SliceRowAsVector(sampleIndex);
    const int yDim = y.GetN();

    ASSERT_ARG_DIM_1D(yDim, stats.GetN()/2);

    counts += weight;

    for(int d=0; d<y.GetN(); d++)
    {
        const FloatType y_i = y.Get(d);
        stats.Incr(feature, threshold, d, weight*y_i);
        stats.Incr(feature, threshold, d+yDim, weight*y_i*y_i);
    }
}


// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class MeanVarianceStatsUpdater
{
public:
    MeanVarianceStatsUpdater(const BufferId& sampleWeightsBufferId,
                            const BufferId& ysBufferId,
                            int yDim);

    BindedMeanVarianceStatsUpdater<FloatType, IntType> Bind(const BufferCollectionStack& readCollection) const;
    int GetDimension() const;

    typedef FloatType Float;
    typedef IntType Int;
    typedef BindedMeanVarianceStatsUpdater<FloatType, IntType> BindedStatUpdater;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mYsBufferId;
    const int mYDimension;
};

template <class FloatType, class IntType>
MeanVarianceStatsUpdater<FloatType, IntType>::MeanVarianceStatsUpdater(const BufferId& sampleWeightsBufferId,
                                                                      const BufferId& ysBufferId,
                                                                      int yDim)
: mSampleWeightsBufferId(sampleWeightsBufferId)
, mYsBufferId(ysBufferId)
, mYDimension(yDim)
{}

template <class FloatType, class IntType>
BindedMeanVarianceStatsUpdater<FloatType, IntType>
MeanVarianceStatsUpdater<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection) const
{
    VectorBufferTemplate<FloatType> const* sampleWeights =
          readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);

    MatrixBufferTemplate<FloatType> const* ys =
          readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mYsBufferId);

    return BindedMeanVarianceStatsUpdater<FloatType, IntType>(sampleWeights, ys);
}

template <class FloatType, class IntType>
int MeanVarianceStatsUpdater<FloatType, IntType>::GetDimension() const
{
    return mYDimension*2;
}

