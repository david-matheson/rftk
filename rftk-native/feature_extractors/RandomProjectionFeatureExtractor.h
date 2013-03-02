#pragma once

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real.hpp>

#include "bootstrap.h"
#include "MatrixBuffer.h"
#include "SparseMatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"

template<typename XFloatBufferType>
class RandomProjectionFeatureExtractor : public FeatureExtractorI {
public:
    RandomProjectionFeatureExtractor(   int numberOfFeatures,
                                        int numberOfComponents,
                                        int numberOfComponentsInSubspace,
                                        bool usePoisson = false );
    ~RandomProjectionFeatureExtractor();
    virtual FeatureExtractorI* Clone() const;

    virtual int GetUID() const { return VEC_FEATURE_PROJECTION; }

    virtual int GetNumberOfFeatures() const;
    virtual Float32MatrixBuffer CreateFloatParams(const int numberOfFeatures) const;
    virtual Int32MatrixBuffer CreateIntParams(const int numberOfFeatures) const;

    virtual int GetFloatParamsDim() const;
    virtual int GetIntParamsDim() const;

    virtual void Extract( const BufferCollection& data,
                            const Int32VectorBuffer& sampleIndices,
                            const Int32MatrixBuffer& intFeatureParams,
                            const Float32MatrixBuffer& floatFeatureParams,
                            Float32MatrixBuffer& featureValuesOUT) const; // #tests X #samples

private:
    int mNumberOfFeatures;
    int mNumberOfComponents;
    int mNumberOfComponentsInSubspace;
    bool mUsePoisson;
    mutable boost::mt19937 mGen;
};




template<typename XFloatBufferType>
RandomProjectionFeatureExtractor<XFloatBufferType>::
RandomProjectionFeatureExtractor(int numberOfFeatures,
                                 int numberOfComponents,
                                 int numberOfComponentsInSubspace,
                                 bool usePoisson )
: mNumberOfFeatures(numberOfFeatures)
, mNumberOfComponents(numberOfComponents)
, mNumberOfComponentsInSubspace(numberOfComponentsInSubspace)
, mUsePoisson(usePoisson)
, mGen( static_cast<unsigned int>(std::time(NULL)) )
{
}

template<typename XFloatBufferType>
RandomProjectionFeatureExtractor<XFloatBufferType>::~RandomProjectionFeatureExtractor()
{}

template<typename XFloatBufferType>
FeatureExtractorI* RandomProjectionFeatureExtractor<XFloatBufferType>::Clone() const
{
    return new RandomProjectionFeatureExtractor(*this);
}

template<typename XFloatBufferType>
int RandomProjectionFeatureExtractor<XFloatBufferType>::GetNumberOfFeatures() const
{
    int numberOfFeatures = mNumberOfFeatures;
    if(mUsePoisson)
    {
        boost::poisson_distribution<> poisson(static_cast<double>(mNumberOfFeatures));
        boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(mGen, poisson);
        numberOfFeatures = std::max(1, var_poisson());
    }
    return numberOfFeatures;
}

template<typename XFloatBufferType>
Float32MatrixBuffer RandomProjectionFeatureExtractor<XFloatBufferType>::CreateFloatParams(const int numberOfFeatures) const
{
    Float32MatrixBuffer floatParams = Float32MatrixBuffer(numberOfFeatures, GetFloatParamsDim());

    boost::uniform_real<float> uniform(-1.0f, 1.0f);
    boost::variate_generator<boost::mt19937&,boost::uniform_real<float> > var_uniform_float(mGen, uniform);

    for(int i=0; i<numberOfFeatures; i++)
    {
        for(int c=0; c<mNumberOfComponentsInSubspace; c++)
        {
            const float componentProjection = var_uniform_float();
            floatParams.Set(i, c+2, componentProjection);
        }
    }
    return floatParams;
}

template<typename XFloatBufferType>
Int32MatrixBuffer RandomProjectionFeatureExtractor<XFloatBufferType>::CreateIntParams(const int numberOfFeatures) const
{
    Int32MatrixBuffer intParams = Int32MatrixBuffer(numberOfFeatures, GetIntParamsDim());
    for(int i=0; i<numberOfFeatures; i++)
    {
        // intParams[0,:] is reserved for feature type
        intParams.Set(i, 0, GetUID());
        intParams.Set(i, 1, mNumberOfComponentsInSubspace);

        std::vector<int> subspace(mNumberOfComponentsInSubspace);
        sampleIndicesWithOutReplacement(&subspace[0], mNumberOfComponentsInSubspace, mNumberOfComponents);
        for(int c=0; c<mNumberOfComponentsInSubspace; c++)
        {
            intParams.Set(i, c+2, subspace[c]);
        }
    }
    return intParams;
}

//Includes the threshold
template<typename XFloatBufferType>
int RandomProjectionFeatureExtractor<XFloatBufferType>::GetFloatParamsDim() const { return mNumberOfComponentsInSubspace + 2; }

//Includes FeatureExtractor id
template<typename XFloatBufferType>
int RandomProjectionFeatureExtractor<XFloatBufferType>::GetIntParamsDim() const { return mNumberOfComponentsInSubspace + 2; }

template<typename XFloatBufferType>
void RandomProjectionFeatureExtractor<XFloatBufferType>
::Extract(const BufferCollection& data,
          const Int32VectorBuffer& sampleIndices,
          const Int32MatrixBuffer& intFeatureParams,
          const Float32MatrixBuffer& floatFeatureParams,
          Float32MatrixBuffer& featureValuesOUT) const// #features X #samples
{
    ASSERT_ARG_DIM_1D(intFeatureParams.GetM(), floatFeatureParams.GetM());
    ASSERT( data.HasBuffer(X_FLOAT_DATA) );
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

        for(int f=0; f<numberOfFeatures; f++)
        {
            float projectionValue = 0.0f;
            const int numberOfComponentsInProjection = intFeatureParams.Get(f, 1);
            for(int c=2; c<numberOfComponentsInProjection+2; c++)
            {
                const int componentId = intFeatureParams.Get(f, c);
                const float componentProjection = floatFeatureParams.Get(f, c);
                projectionValue += Xs.Get(index, componentId) * componentProjection;
            }
            featureValuesOUT.Set(s, f, projectionValue);
        }
    }
}


/**
 * Interface for swig
 */
#define DECLARE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(NAME)  \
typedef RandomProjectionFeatureExtractor<NAME ## MatrixBuffer> \
    NAME ## RandomProjectionFeatureExtractorType; \
NAME ## RandomProjectionFeatureExtractorType NAME ## RandomProjectionFeatureExtractor \
    (int numberOfFeatures, \
     int numberOfComponents, \
     int numberOfComponentsInSubspace, \
     bool usePoisson = false );

DECLARE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float32);
DECLARE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float64);
DECLARE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float32Sparse);
DECLARE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float64Sparse);

#undef DECLARE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY
