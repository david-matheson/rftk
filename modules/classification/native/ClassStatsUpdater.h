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
class BindedClassStatsUpdater
{
public:
    BindedClassStatsUpdater(VectorBufferTemplate<FloatType> const* sampleWeights,
                            VectorBufferTemplate<IntType> const* mClasses);

    void UpdateStats(FloatType& counts, Tensor3BufferTemplate<FloatType>& stats,
                int feature, int threshold, int sampleIndex) const;


private:
    VectorBufferTemplate<FloatType> const* mSampleWeights;
    VectorBufferTemplate<IntType> const* mClasses;
};

template <class FloatType, class IntType>
BindedClassStatsUpdater<FloatType, IntType>::BindedClassStatsUpdater(VectorBufferTemplate<FloatType> const* sampleWeights,
                                                                      VectorBufferTemplate<IntType> const* classes)
: mSampleWeights(sampleWeights)
, mClasses(classes)
{}

template <class FloatType, class IntType>
void BindedClassStatsUpdater<FloatType, IntType>::UpdateStats(FloatType& counts, Tensor3BufferTemplate<FloatType>& stats,
                                                          int feature, int threshold, int sampleIndex) const
{
    const FloatType weight = mSampleWeights->Get(sampleIndex);
    const IntType classId = mClasses->Get(sampleIndex);

    counts += weight;
    stats.Incr(feature, threshold, classId, weight);
}


// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class ClassStatsUpdater
{
public:
    ClassStatsUpdater(const BufferId& sampleWeightsBufferId,
                       const BufferId& classesBufferId,
                       const int numberOfClasses);

    BindedClassStatsUpdater<FloatType, IntType> Bind(const BufferCollectionStack& readCollection) const;
    int GetDimension() const;

    typedef FloatType Float;
    typedef IntType Int;
    typedef BindedClassStatsUpdater<FloatType, IntType> BindedStatUpdater;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mClassesBufferId;
    const int mNumberOfClasses;
};

template <class FloatType, class IntType>
ClassStatsUpdater<FloatType, IntType>::ClassStatsUpdater(const BufferId& sampleWeightsBufferId,
                                                          const BufferId& classesBufferId,
                                                          const int numberOfClasses)
: mSampleWeightsBufferId(sampleWeightsBufferId)
, mClassesBufferId(classesBufferId)
, mNumberOfClasses(numberOfClasses)
{}

template <class FloatType, class IntType>
BindedClassStatsUpdater<FloatType, IntType>
ClassStatsUpdater<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection) const
{
    VectorBufferTemplate<FloatType> const* sampleWeights =
          readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);

    VectorBufferTemplate<IntType> const* classes =
          readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mClassesBufferId);

    return BindedClassStatsUpdater<FloatType, IntType>(sampleWeights, classes);
}

template <class FloatType, class IntType>
int ClassStatsUpdater<FloatType, IntType>::GetDimension() const
{
    return mNumberOfClasses;
}
