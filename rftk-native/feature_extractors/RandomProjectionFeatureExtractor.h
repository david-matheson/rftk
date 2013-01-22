#pragma once

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

    virtual int GetUID() const { return VEC_FEATURE_PROJECTION; }

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
    int mNumberOfComponentsInSubspace;
    bool mUsePoisson;
};
