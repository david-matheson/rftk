#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

class FeatureExtractorI {
public:
    virtual FeatureExtractorI* Clone() const=0;

    virtual void Extract(const BufferCollection& data,
                        const MatrixBufferInt& sampleIndices,
                        const MatrixBufferInt& intFeatureParams,
                        const MatrixBufferFloat& floatFeatureParams,
                        MatrixBufferFloat& featureValuesOUT ) const=0;// #tests X #samples

    virtual int GetUID() const=0;
    virtual int GetNumberOfFeatures() const=0;
    virtual MatrixBufferFloat CreateFloatParams(const int numberOfFeatures) const=0;
    virtual MatrixBufferInt CreateIntParams(const int numberOfFeatures) const=0;

    virtual int GetFloatParamsDim() const=0;
    virtual int GetIntParamsDim() const=0;
};