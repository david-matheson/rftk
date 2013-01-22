#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"


class AxisAlignedFeatureExtractor : public FeatureExtractorI {
public:
    AxisAlignedFeatureExtractor(int numberOfFeatures, int numberOfComponents, bool usePoisson = false);

    ~AxisAlignedFeatureExtractor();

    virtual int GetUID() const { return VEC_FEATURE_AXIS_ALIGNED; }

    virtual int GetNumberOfFeatures() const;
    virtual MatrixBufferFloat CreateFloatParams(const int numberOfFeatures) const;
    virtual MatrixBufferInt CreateIntParams(const int numberOfFeatures) const;

    virtual int GetFloatParamsDim() const;
    virtual int GetIntParamsDim() const;

    virtual void Extract( BufferCollection& data,
                            const MatrixBufferInt& sampleIndices,
                            const MatrixBufferInt& intFeatureParams,
                            const MatrixBufferFloat& floatFeatureParams,
                            MatrixBufferFloat& featureValuesOUT) const; // #tests X #samples

private:
    int mNumberOfFeatures;
    int mNumberOfComponents;
    bool mUsePoisson;
    mutable boost::mt19937 mGen;
};
