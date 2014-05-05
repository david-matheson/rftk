#pragma once

#include <VectorBuffer.h>
#include <MatrixBuffer.h>
#include <BufferCollectionStack.h>
#include <UniqueBufferId.h>
#include <Tree.h>

// ----------------------------------------------------------------------------
//
// Update estimator sufficient stats from sample weights and ys
//
// ----------------------------------------------------------------------------
template <class BT>
class BindedMeanVarianceEstimatorUpdater
{
public:
    BindedMeanVarianceEstimatorUpdater();

    BindedMeanVarianceEstimatorUpdater(VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                            MatrixBufferTemplate<typename BT::SourceContinuous> const* ys);

    void UpdateEstimator(Tree& tree, 
                              const int nodeIndex,
                              int sampleIndex ) const;

private:
    VectorBufferTemplate<typename BT::DatapointCounts> const* mSampleWeights;
    MatrixBufferTemplate<typename BT::SourceContinuous> const* mYs;
};

template <class BT>
BindedMeanVarianceEstimatorUpdater<BT>::BindedMeanVarianceEstimatorUpdater()
: mSampleWeights(NULL)
, mYs(NULL)
{}


template <class BT>
BindedMeanVarianceEstimatorUpdater<BT>::BindedMeanVarianceEstimatorUpdater(
                                                              VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                                                              MatrixBufferTemplate<typename BT::SourceContinuous> const* ys)
: mSampleWeights(sampleWeights)
, mYs(ys)
{}

template <class BT>
void BindedMeanVarianceEstimatorUpdater<BT>::UpdateEstimator(Tree& tree, 
                                                                const int nodeIndex,
                                                                int sampleIndex ) const
{
    VectorBufferTemplate<float>& treeCounts = tree.GetCounts();
    MatrixBufferTemplate<float>& treeYs = tree.GetYs();

    const typename BT::DatapointCounts counts = treeCounts.Get(nodeIndex);
    const typename BT::DatapointCounts weight = mSampleWeights->Get(sampleIndex);
    const typename BT::DatapointCounts newCounts = weight + counts;
    const typename BT::Index yDim = mYs->GetN();

    for(typename BT::Index d=0; d<yDim; d++)
    {
        const typename BT::SufficientStatsContinuous y_i = mYs->Get(sampleIndex, d);
        // old unstable sufficient stats 
        // stats.Incr(feature, threshold, d, weight*y_i);
        // stats.Incr(feature, threshold, d+yDim, weight*y_i*y_i);
        const typename BT::SufficientStatsContinuous mean = treeYs.Get(nodeIndex, d);
        const typename BT::SufficientStatsContinuous delta = y_i - mean;
        const typename BT::SufficientStatsContinuous r = delta * weight / newCounts;
        treeYs.Incr(nodeIndex, d, r);
        treeYs.Incr(nodeIndex, d+yDim, counts*delta*r);
    }
}


// ----------------------------------------------------------------------------
//
// Update estimator sufficient stats from sample weights and ys
//
// ----------------------------------------------------------------------------
template <class BT>
class MeanVarianceEstimatorUpdater
{
public:
    MeanVarianceEstimatorUpdater(const BufferId& sampleWeightsBufferId,
                       const BufferId& ysBufferId);

    BindedMeanVarianceEstimatorUpdater<BT> Bind(const BufferCollectionStack& readCollection) const;

    typedef BindedMeanVarianceEstimatorUpdater<BT> BindedEstimatorUpdater;
    typedef BT BufferTypes;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mYsBufferId;
};

template <class BT>
MeanVarianceEstimatorUpdater<BT>::MeanVarianceEstimatorUpdater(const BufferId& sampleWeightsBufferId,
                                                          const BufferId& ysBufferId)
: mSampleWeightsBufferId(sampleWeightsBufferId)
, mYsBufferId(ysBufferId)
{}

template <class BT>
BindedMeanVarianceEstimatorUpdater<BT> 
MeanVarianceEstimatorUpdater<BT>::Bind(const BufferCollectionStack& readCollection) const
{
    VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights = 
          readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::DatapointCounts> >(mSampleWeightsBufferId);

    MatrixBufferTemplate<typename BT::SourceContinuous> const* ys = 
          readCollection.GetBufferPtr< MatrixBufferTemplate<typename BT::SourceContinuous> >(mYsBufferId);

    return BindedMeanVarianceEstimatorUpdater<BT>(sampleWeights, ys);
}

