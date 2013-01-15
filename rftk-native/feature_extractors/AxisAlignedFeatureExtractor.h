#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"


class AxisAlignedFeatureExtractor : public FeatureExtractorI {
public:
    AxisAlignedFeatureExtractor(int numberOfFeatures, int numberOfComponents);

    ~AxisAlignedFeatureExtractor();

    virtual int GetUID() const { return VEC_FEATURE_AXIS_ALIGNED; }

    virtual MatrixBufferFloat CreateFloatParams() const;
    virtual MatrixBufferInt CreateIntParams() const;

    virtual int GetFloatParamsDim() const;
    virtual int GetIntParamsDim() const;

    virtual void Extract( BufferCollection& data,
                            const MatrixBufferInt& sampleIndices,
                            const MatrixBufferInt& intFeatureParams,
                            const MatrixBufferFloat& floatFeatureParams,
                            MatrixBufferFloat& featureValuesOUT); // #tests X #samples

private:
    int mNumberOfFeatures;
    int mNumberOfComponents;
};
