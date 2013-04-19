#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

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
class FeatureExtractorStep: public PipelineStepI
{
public:
    FeatureExtractorStep(const FeatureType& feature, FeatureValueOrdering ordering);
    virtual ~FeatureExtractorStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

    // Read only output buffer
    const UniqueBufferId::BufferId FeatureValuesBufferId;
private:
    const FeatureType mFeature;
    FeatureValueOrdering mOrdering;
};

template <class FeatureType>
FeatureExtractorStep<FeatureType>::FeatureExtractorStep(const FeatureType& feature, FeatureValueOrdering ordering)
: FeatureValuesBufferId(UniqueBufferId::GetBufferId("FeatureValues"))
, mFeature(feature)
, mOrdering(ordering)
{}

template <class FeatureType>
FeatureExtractorStep<FeatureType>::~FeatureExtractorStep()
{}

template <class FeatureType>
PipelineStepI* FeatureExtractorStep<FeatureType>::Clone() const
{
    FeatureExtractorStep* clone = new FeatureExtractorStep<FeatureType>(*this);
    return clone;
}

template <class FeatureType>
void FeatureExtractorStep<FeatureType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                BufferCollection& writeCollection) const
{
    FeatureType feature = mFeature;
    feature.Bind(readCollection);

    typename FeatureType::Int numberOfFeatures = feature.GetNumberOfFeatures();
    typename FeatureType::Int numberOfDatapoints = feature.GetNumberOfDatapoints();

    typename FeatureType::Int m = (mOrdering == FEATURES_BY_DATAPOINTS) ? numberOfFeatures : numberOfDatapoints;
    typename FeatureType::Int n = (mOrdering == FEATURES_BY_DATAPOINTS) ? numberOfDatapoints : numberOfFeatures;

    MatrixBufferTemplate<typename FeatureType::Float>& featureValues =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename FeatureType::Float> >(FeatureValuesBufferId);
    featureValues.Resize(m,n);

    for(int s=0; s<numberOfDatapoints; s++)
    {
        for(int f=0; f<numberOfFeatures; f++)
        {
            typename FeatureType::Float value = feature.FeatureValue(f, s);
            typename FeatureType::Int r = (mOrdering == FEATURES_BY_DATAPOINTS) ? f : s;
            typename FeatureType::Int c = (mOrdering == FEATURES_BY_DATAPOINTS) ? s : f;
            featureValues.Set(r,c, value);
        }
    }
}