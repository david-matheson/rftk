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
    virtual FeatureExtractorI* Clone() const;

    virtual int GetUID() const { return VEC_FEATURE_AXIS_ALIGNED; }

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
    bool mUsePoisson;
    mutable boost::mt19937 mGen;
};
