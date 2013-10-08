#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "FeatureOrdering.h"


template <class BufferTypes>
class FeatureRangeStep: public PipelineStepI
{
public:
    FeatureRangeStep( const BufferId& featureValuesBufferId,
                        FeatureValueOrdering ordering );
    virtual ~FeatureRangeStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    const BufferId FeatureRangeMinMaxBufferId;
private:

    const BufferId mFeatureValuesBufferId;
    FeatureValueOrdering mOrdering;
};


template <class BufferTypes>
FeatureRangeStep<BufferTypes>::FeatureRangeStep(  const BufferId& featureValuesBufferId,
                                                    FeatureValueOrdering ordering )
: PipelineStepI("FeatureRangeStep")
, FeatureRangeMinMaxBufferId(GetBufferId("FeatureRangeMinMax"))
, mFeatureValuesBufferId(featureValuesBufferId)
, mOrdering(ordering)
{}

template <class BufferTypes>
FeatureRangeStep<BufferTypes>::~FeatureRangeStep()
{}

template <class BufferTypes>
PipelineStepI* FeatureRangeStep<BufferTypes>::Clone() const
{
    FeatureRangeStep* clone = new FeatureRangeStep<BufferTypes>(*this);
    return clone;
}

template <class BufferTypes>
void FeatureRangeStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen,
                                                          BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& featureValues =
            readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(mFeatureValuesBufferId);

    typename BufferTypes::Index numberOfFeatures = (mOrdering == FEATURES_BY_DATAPOINTS) ? featureValues.GetM() : featureValues.GetN();
    typename BufferTypes::Index numberOfDatapoints = (mOrdering == FEATURES_BY_DATAPOINTS) ? featureValues.GetN() : featureValues.GetM();

    MatrixBufferTemplate<typename BufferTypes::FeatureValue>& minMaxValues =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(FeatureRangeMinMaxBufferId);

    minMaxValues.Resize(numberOfFeatures, 2);

    for(int f=0; f<numberOfFeatures; f++)
    {
        for(int s=0; s<numberOfDatapoints; s++)
        {
            typename BufferTypes::Index r = (mOrdering == FEATURES_BY_DATAPOINTS) ? f : s;
            typename BufferTypes::Index c = (mOrdering == FEATURES_BY_DATAPOINTS) ? s : f;
            typename BufferTypes::FeatureValue value = featureValues.Get(r, c);

            typename BufferTypes::FeatureValue min = minMaxValues.Get(f,0);
            typename BufferTypes::FeatureValue max = minMaxValues.Get(f,1);

            if( value < min || s == 0)
            {
                minMaxValues.Set(f, 0, value);
            }

            if( value > max || s == 0)
            {
                minMaxValues.Set(f, 1, value);
            }
        }
    }    
}


