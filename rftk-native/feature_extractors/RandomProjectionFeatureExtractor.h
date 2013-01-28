#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"


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
                            const Int32MatrixBuffer& sampleIndices,
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
