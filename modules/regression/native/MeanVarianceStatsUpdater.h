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
template <class BT>
class BindedMeanVarianceStatsUpdater
{
public:
    BindedMeanVarianceStatsUpdater(VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                            MatrixBufferTemplate<typename BT::SourceContinuous> const* ys);

    void UpdateStats(typename BT::DatapointCounts& counts, Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& stats,
                int feature, int threshold, int sampleIndex) const;

private:
    VectorBufferTemplate<typename BT::DatapointCounts> const* mSampleWeights;
    MatrixBufferTemplate<typename BT::SourceContinuous> const* mYs;
};

template <class BT>
BindedMeanVarianceStatsUpdater<BT>::BindedMeanVarianceStatsUpdater(VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                                                                                  MatrixBufferTemplate<typename BT::SourceContinuous> const* ys)
: mSampleWeights(sampleWeights)
, mYs(ys)
{}

template <class BT>
void BindedMeanVarianceStatsUpdater<BT>::UpdateStats(typename BT::DatapointCounts& counts, Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& stats,
                                                          int feature, int threshold, int sampleIndex) const
{
    const typename BT::DatapointCounts weight = mSampleWeights->Get(sampleIndex);
    const VectorBufferTemplate<typename BT::SourceContinuous> y = mYs->SliceRowAsVector(sampleIndex);
    const typename BT::Index yDim = y.GetN();

    ASSERT_ARG_DIM_1D(yDim, stats.GetN()/2);

    const typename BT::DatapointCounts newCounts = weight + counts;
    for(typename BT::Index d=0; d<y.GetN(); d++)
    {
        const typename BT::SufficientStatsContinuous y_i = y.Get(d);
        // old unstable sufficient stats 
        // stats.Incr(feature, threshold, d, weight*y_i);
        // stats.Incr(feature, threshold, d+yDim, weight*y_i*y_i);
        const typename BT::SufficientStatsContinuous mean = stats.Get(feature, threshold, d);
        const typename BT::SufficientStatsContinuous delta = y_i - mean;
        const typename BT::SufficientStatsContinuous r = delta * weight / newCounts;
        stats.Incr(feature, threshold, d, r);
        stats.Incr(feature, threshold, d+yDim, counts*delta*r);
    }
    counts = newCounts;
}


// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class BT>
class MeanVarianceStatsUpdater
{
public:
    MeanVarianceStatsUpdater(const BufferId& sampleWeightsBufferId,
                            const BufferId& ysBufferId,
                            int yDim);

    BindedMeanVarianceStatsUpdater<BT> Bind(const BufferCollectionStack& readCollection) const;
    int GetDimension() const;

    typedef BindedMeanVarianceStatsUpdater<BT> BindedStatUpdater;
    typedef BT BufferTypes;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mYsBufferId;
    const int mYDimension;
};

template <class BT>
MeanVarianceStatsUpdater<BT>::MeanVarianceStatsUpdater(const BufferId& sampleWeightsBufferId,
                                                                      const BufferId& ysBufferId,
                                                                      int yDim)
: mSampleWeightsBufferId(sampleWeightsBufferId)
, mYsBufferId(ysBufferId)
, mYDimension(yDim)
{}

template <class BT>
BindedMeanVarianceStatsUpdater<BT>
MeanVarianceStatsUpdater<BT>::Bind(const BufferCollectionStack& readCollection) const
{
    VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights =
          readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::DatapointCounts> >(mSampleWeightsBufferId);

    MatrixBufferTemplate<typename BT::SourceContinuous> const* ys =
          readCollection.GetBufferPtr< MatrixBufferTemplate<typename BT::SourceContinuous> >(mYsBufferId);

    return BindedMeanVarianceStatsUpdater<BT>(sampleWeights, ys);
}

template <class BT>
int MeanVarianceStatsUpdater<BT>::GetDimension() const
{
    return mYDimension*2;
}

