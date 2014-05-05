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
class BindedClassStatsUpdater
{
public:
    BindedClassStatsUpdater(VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                            VectorBufferTemplate<typename BT::SourceInteger> const* mClasses);

    void UpdateStats(typename BT::DatapointCounts& counts, Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& stats,
                int feature, int threshold, int sampleIndex) const;

private:
    VectorBufferTemplate<typename BT::DatapointCounts> const* mSampleWeights;
    VectorBufferTemplate<typename BT::SourceInteger> const* mClasses;
};

template <class BT>
BindedClassStatsUpdater<BT>::BindedClassStatsUpdater(VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                                                                      VectorBufferTemplate<typename BT::SourceInteger> const* classes)
: mSampleWeights(sampleWeights)
, mClasses(classes)
{}

template <class BT>
void BindedClassStatsUpdater<BT>::UpdateStats(typename BT::DatapointCounts& counts, Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& stats,
                                                          int feature, int threshold, int sampleIndex) const
{
    const typename BT::DatapointCounts weight = mSampleWeights->Get(sampleIndex);
    const typename BT::SourceInteger classId = mClasses->Get(sampleIndex);

    counts += weight;
    stats.Incr(feature, threshold, classId, weight);
}


// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class BT>
class ClassStatsUpdater
{
public:
    ClassStatsUpdater(const BufferId& sampleWeightsBufferId,
                       const BufferId& classesBufferId,
                       const int numberOfClasses);

    BindedClassStatsUpdater<BT> Bind(const BufferCollectionStack& readCollection) const;
    int GetDimension() const;

    typedef BindedClassStatsUpdater<BT> BindedStatUpdater;
    typedef BT BufferTypes;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mClassesBufferId;
    const int mNumberOfClasses;
};

template <class BT>
ClassStatsUpdater<BT>::ClassStatsUpdater(const BufferId& sampleWeightsBufferId,
                                                          const BufferId& classesBufferId,
                                                          const int numberOfClasses)
: mSampleWeightsBufferId(sampleWeightsBufferId)
, mClassesBufferId(classesBufferId)
, mNumberOfClasses(numberOfClasses)
{}

template <class BT>
BindedClassStatsUpdater<BT>
ClassStatsUpdater<BT>::Bind(const BufferCollectionStack& readCollection) const
{
    VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights =
          readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::DatapointCounts> >(mSampleWeightsBufferId);

    VectorBufferTemplate<typename BT::SourceInteger> const* classes =
          readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::SourceInteger> >(mClassesBufferId);

    return BindedClassStatsUpdater<BT>(sampleWeights, classes);
}

template <class BT>
int ClassStatsUpdater<BT>::GetDimension() const
{
    return mNumberOfClasses;
}
