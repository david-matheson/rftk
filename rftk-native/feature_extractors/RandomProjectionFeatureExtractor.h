#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"


class RandomProjectionFeatureExtractor : public FeatureExtractorI {
public:
    RandomProjectionFeatureExtractor(   int numberOfFeatures, 
                                        int numberOfComponents,
                                        int numberOfComponentsInSubspace );

    ~RandomProjectionFeatureExtractor();

    virtual int GetUID() const { return VEC_FEATURE_PROJECTION; }

    virtual MatrixBufferFloat CreateFloatParams() const;
    virtual MatrixBufferInt CreateIntParams() const;

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
    int mNumberOfComponentsInSubspace;

};
