#pragma once

#include "buffers/MatrixBuffer.h"
#include "buffers/BufferCollection.h"

class FeatureExtractorI {
public:
    virtual ~FeatureExtractorI() {}
    virtual FeatureExtractorI* Clone() const=0;

    virtual void Extract(const BufferCollection& data,
                        const Int32VectorBuffer& sampleIndices,
                        const Int32MatrixBuffer& intFeatureParams,
                        const Float32MatrixBuffer& floatFeatureParams,
                        Float32MatrixBuffer& featureValuesOUT ) const=0;// #tests X #samples

    virtual int GetUID() const=0;
    virtual int GetNumberOfFeatures() const=0;
    virtual Float32MatrixBuffer CreateFloatParams(const int numberOfFeatures) const=0;
    virtual Int32MatrixBuffer CreateIntParams(const int numberOfFeatures) const=0;

    virtual int GetFloatParamsDim() const=0;
    virtual int GetIntParamsDim() const=0;
};