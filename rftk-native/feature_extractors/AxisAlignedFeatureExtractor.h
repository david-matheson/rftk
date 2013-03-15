#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <ctime>

#include "bootstrap.h"
#include "MatrixBuffer.h"
#include "SparseMatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"

template<class XFloatBufferType>
class AxisAlignedFeatureExtractor : public FeatureExtractorI {
public:
    AxisAlignedFeatureExtractor(int numberOfFeatures, int numberOfComponents, bool usePoisson = false);
    virtual ~AxisAlignedFeatureExtractor() { }
    virtual FeatureExtractorI* Clone() const;

    virtual int GetUID() const { return VEC_FEATURE_AXIS_ALIGNED; }

    virtual int GetNumberOfFeatures() const;
    virtual Float32MatrixBuffer CreateFloatParams(const int numberOfFeatures) const;
    virtual Int32MatrixBuffer CreateIntParams(const int numberOfFeatures) const;

    virtual int GetFloatParamsDim() const;
    virtual int GetIntParamsDim() const;


    virtual void Extract(const BufferCollection& data,
                         const Int32VectorBuffer& sampleIndices,
                         const Int32MatrixBuffer& intFeatureParams,
                         const Float32MatrixBuffer& floatFeatureParams,
                         Float32MatrixBuffer& featureValuesOUT) const; // #tests X #samples

private:
    int mNumberOfFeatures;
    int mNumberOfComponents;
    bool mUsePoisson;
    mutable boost::mt19937 mGen;
};

template<typename XFloatBufferType>
AxisAlignedFeatureExtractor<XFloatBufferType>::AxisAlignedFeatureExtractor(int numberOfFeatures, int numberOfComponents, bool usePoisson)
  : mNumberOfFeatures(numberOfFeatures)
  , mNumberOfComponents(numberOfComponents)
  , mUsePoisson(usePoisson)
  , mGen( static_cast<unsigned int>(std::time(NULL)) )
{
}

template<typename XFloatBufferType>
FeatureExtractorI* AxisAlignedFeatureExtractor<XFloatBufferType>::Clone() const
{
    return new AxisAlignedFeatureExtractor<XFloatBufferType>(*this);
}

template<typename XFloatBufferType>
int AxisAlignedFeatureExtractor<XFloatBufferType>::GetNumberOfFeatures() const
{
    int numberOfFeatures = mNumberOfFeatures;
    if(mUsePoisson)
    {
        boost::poisson_distribution<> poisson(static_cast<double>(mNumberOfFeatures));
        boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(mGen, poisson);
        numberOfFeatures = std::min(mNumberOfComponents, std::max(1, var_poisson())); //can't have more features than components
    }
    return numberOfFeatures;
}

template<typename XFloatBufferType>
Float32MatrixBuffer AxisAlignedFeatureExtractor<XFloatBufferType>::CreateFloatParams(const int numberOfFeatures) const
{
    Float32MatrixBuffer floatParams = Float32MatrixBuffer(numberOfFeatures, GetFloatParamsDim());
    return floatParams;
}

template<typename XFloatBufferType>
Int32MatrixBuffer AxisAlignedFeatureExtractor<XFloatBufferType>::CreateIntParams(const int numberOfFeatures) const
{
    Int32MatrixBuffer intParams = Int32MatrixBuffer(numberOfFeatures, GetIntParamsDim());
    std::vector<int> axes(numberOfFeatures);
    sampleIndicesWithOutReplacement(&axes[0], numberOfFeatures, mNumberOfComponents);

    for(int i=0; i<numberOfFeatures; i++)
    {
        intParams.Set(i, 0, GetUID());
        intParams.Set(i, 1, axes[i]);
    }
    return intParams;
}

//Includes the threshold
template<typename XFloatBufferType>
int AxisAlignedFeatureExtractor<XFloatBufferType>::GetFloatParamsDim() const { return 1; }

//Includes FeatureExtractor id
template<typename XFloatBufferType>
int AxisAlignedFeatureExtractor<XFloatBufferType>::GetIntParamsDim() const { return 2; }

template<typename XFloatBufferType>
void AxisAlignedFeatureExtractor<XFloatBufferType>::
Extract(const BufferCollection& data,
        const Int32VectorBuffer& sampleIndices,
        const Int32MatrixBuffer& intFeatureParams,
        const Float32MatrixBuffer& floatFeatureParams,
        Float32MatrixBuffer& featureValuesOUT) const// #features X #samples
{
    UNUSED_PARAM(floatFeatureParams);
    ASSERT_ARG_DIM_1D(intFeatureParams.GetM(), floatFeatureParams.GetM());
    ASSERT(data.HasBuffer(X_FLOAT_DATA));
    ASSERT_ARG_DIM_1D(data.GetBuffer<XFloatBufferType>(X_FLOAT_DATA).GetN(), mNumberOfComponents);

    const XFloatBufferType& Xs = data.GetBuffer<XFloatBufferType>(X_FLOAT_DATA);

    const int numberOfSamples = sampleIndices.GetN();
    const int numberOfFeatures = intFeatureParams.GetM();

    featureValuesOUT.Resize(numberOfSamples, numberOfFeatures);

    // Extract component features
    for(int s=0; s<numberOfSamples; s++)
    {
        const int index = sampleIndices.Get(s);
        ASSERT_VALID_RANGE(index, 0, Xs.GetM());

        for(int t=0; t<numberOfFeatures; t++)
        {
            const int componentId = intFeatureParams.Get(t, 1);
            const float value =  Xs.Get(index, componentId);
            featureValuesOUT.Set(s, t, value);
        }
    }
}


/**
 * Interface for swig
 */
#define DECLARE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(NAME)  \
typedef AxisAlignedFeatureExtractor<NAME ## MatrixBuffer> \
    NAME ## AxisAlignedFeatureExtractorType; \
NAME ## AxisAlignedFeatureExtractorType NAME ## AxisAlignedFeatureExtractor \
    (int numberOfFeatures, \
     int numberOfComponents, \
     bool usePoisson=false);

DECLARE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float32);
DECLARE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float64);
DECLARE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float32Sparse);
DECLARE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float64Sparse);

#undef DECLARE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY
