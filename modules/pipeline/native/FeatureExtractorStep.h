#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "FeatureIndexerI.h"

enum FeatureValueOrdering
{
    FEATURES_BY_DATAPOINTS,
    DATAPOINTS_BY_FEATURES
};


// ----------------------------------------------------------------------------
//
// FeatureExtractorStep extracts features for all float/int params for all
// datapoints
//
// ----------------------------------------------------------------------------
template <class FeatureType>
class FeatureExtractorStep: public PipelineStepI, public FeatureIndexerI
{
public:
    FeatureExtractorStep(const FeatureType& feature, FeatureValueOrdering ordering);
    virtual ~FeatureExtractorStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    virtual int IndexFeature( const BufferCollectionStack& readCollection,
                              const int featureOffset  ) const;

    virtual FeatureIndexerI* CloneFeatureIndexerI() const;

    // Read only output buffer
    const BufferId FeatureValuesBufferId;
private:
    const FeatureType mFeature;
    FeatureValueOrdering mOrdering;
};

template <class FeatureType>
FeatureExtractorStep<FeatureType>::FeatureExtractorStep(const FeatureType& feature, FeatureValueOrdering ordering)
: PipelineStepI("FeatureExtractorStep")
, FeatureValuesBufferId(GetBufferId("FeatureValues"))
, mFeature(feature)
, mOrdering(ordering)
{}

template <class FeatureType>
FeatureExtractorStep<FeatureType>::~FeatureExtractorStep()
{}

template <class FeatureType>
PipelineStepI* FeatureExtractorStep<FeatureType>::Clone() const
{
    PipelineStepI* clone = new FeatureExtractorStep<FeatureType>(*this);
    return clone;
}

template <class FeatureType>
void FeatureExtractorStep<FeatureType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                BufferCollection& writeCollection,
                                                boost::mt19937& gen,
                                                BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);
    typename FeatureType::FeatureBinding featureBinding = mFeature.Bind(readCollection);

    typename FeatureType::Int numberOfFeatures = featureBinding.GetNumberOfFeatures();
    typename FeatureType::Int numberOfDatapoints = featureBinding.GetNumberOfDatapoints();

    typename FeatureType::Int m = (mOrdering == FEATURES_BY_DATAPOINTS) ? numberOfFeatures : numberOfDatapoints;
    typename FeatureType::Int n = (mOrdering == FEATURES_BY_DATAPOINTS) ? numberOfDatapoints : numberOfFeatures;

    MatrixBufferTemplate<typename FeatureType::Float>& featureValues =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename FeatureType::Float> >(FeatureValuesBufferId);
    featureValues.Resize(m,n);

    for(int s=0; s<numberOfDatapoints; s++)
    {
        for(int f=0; f<numberOfFeatures; f++)
        {
            typename FeatureType::Float value = featureBinding.FeatureValue(f, s);
            typename FeatureType::Int r = (mOrdering == FEATURES_BY_DATAPOINTS) ? f : s;
            typename FeatureType::Int c = (mOrdering == FEATURES_BY_DATAPOINTS) ? s : f;
            featureValues.Set(r,c, value);
        }
    }
}

template <class FeatureType>
int FeatureExtractorStep<FeatureType>::IndexFeature(  const BufferCollectionStack& readCollection,
                                                      const int featureOffset  ) const
{
    return mFeature.FeatureIndex(readCollection, featureOffset);
}

template <class FeatureType>
FeatureIndexerI* FeatureExtractorStep<FeatureType>::CloneFeatureIndexerI() const
{
    FeatureIndexerI* clone = new FeatureExtractorStep<FeatureType>(*this);
    return clone;
}