#pragma once

#include <VectorBuffer.h>
#include <MatrixBuffer.h>
#include <BufferCollectionStack.h>
#include <UniqueBufferId.h>
#include <Tree.h>

// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class BT>
class BindedClassEstimatorUpdater
{
public:
    BindedClassEstimatorUpdater();

    BindedClassEstimatorUpdater(VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                            VectorBufferTemplate<typename BT::SourceInteger> const* mClasses,
                            int numberOfClasses);

    void UpdateEstimator(Tree& tree, 
                              const int nodeIndex,
                              int sampleIndex ) const;

private:
    VectorBufferTemplate<typename BT::DatapointCounts> const* mSampleWeights;
    VectorBufferTemplate<typename BT::SourceInteger> const* mClasses;
    int mNumberOfClasses;
};

template <class BT>
BindedClassEstimatorUpdater<BT>::BindedClassEstimatorUpdater()
: mSampleWeights(NULL)
, mClasses(NULL)
, mNumberOfClasses(0)
{}


template <class BT>
BindedClassEstimatorUpdater<BT>::BindedClassEstimatorUpdater(
                                                              VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights,
                                                              VectorBufferTemplate<typename BT::SourceInteger> const* classes,
                                                              int numberOfClasses)
: mSampleWeights(sampleWeights)
, mClasses(classes)
, mNumberOfClasses(numberOfClasses)
{}

template <class BT>
void BindedClassEstimatorUpdater<BT>::UpdateEstimator(Tree& tree, 
                                                                      const int nodeIndex,
                                                                      int sampleIndex ) const
{
    const typename BT::DatapointCounts weight = mSampleWeights->Get(sampleIndex);
    const typename BT::SourceInteger classId = mClasses->Get(sampleIndex);

    const float oldN = tree.GetCounts().Get(nodeIndex);
    for(int c=0; c<mNumberOfClasses; c++)
    {
        float classCount = oldN * tree.GetYs().Get(nodeIndex,c);
        if( classId == c )
        {
            classCount += weight;
        }
        const float updatedClassProbability = classCount / (oldN + weight);
        tree.GetYs().Set(nodeIndex, c, updatedClassProbability);
    }
}


// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class BT>
class ClassEstimatorUpdater
{
public:
    ClassEstimatorUpdater(const BufferId& sampleWeightsBufferId,
                       const BufferId& classesBufferId,
                       const int numberOfClasses);

    BindedClassEstimatorUpdater<BT> Bind(const BufferCollectionStack& readCollection) const;
    int GetNumberOfClasses() const;

    typedef BindedClassEstimatorUpdater<BT> BindedEstimatorUpdater;
    typedef BT BufferTypes;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mClassesBufferId;
    const int mNumberOfClasses;
};

template <class BT>
ClassEstimatorUpdater<BT>::ClassEstimatorUpdater(const BufferId& sampleWeightsBufferId,
                                                          const BufferId& classesBufferId,
                                                          const int numberOfClasses)
: mSampleWeightsBufferId(sampleWeightsBufferId)
, mClassesBufferId(classesBufferId)
, mNumberOfClasses(numberOfClasses)
{}

template <class BT>
BindedClassEstimatorUpdater<BT> 
ClassEstimatorUpdater<BT>::Bind(const BufferCollectionStack& readCollection) const
{
    VectorBufferTemplate<typename BT::DatapointCounts> const* sampleWeights = 
          readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::DatapointCounts> >(mSampleWeightsBufferId);

    VectorBufferTemplate<typename BT::SourceInteger> const* classes = 
          readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::SourceInteger> >(mClassesBufferId);

    return BindedClassEstimatorUpdater<BT>(sampleWeights, classes, mNumberOfClasses);
}

template <class BT>
int ClassEstimatorUpdater<BT>::GetNumberOfClasses() const
{
    return mNumberOfClasses;
}
