#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <MatrixBuffer.h>
#include <BufferCollection.h>
#include <FeatureTypes.h>

#include "FeatureExtractorI.h"



class DepthScaledDepthDeltaFeatureExtractor : public FeatureExtractorI {
public:
    DepthScaledDepthDeltaFeatureExtractor(float sigmaX, float sigmaY, int numberOfFeatures, bool usePoisson );
    ~DepthScaledDepthDeltaFeatureExtractor();
    virtual FeatureExtractorI* Clone() const;

    virtual int GetUID() const { return IMG_FEATURE_DEPTH_DELTA; }

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
    float mSigmaX;
    float mSigmaY;
    int mNumberOfFeatures;
    bool mUsePoisson;
    mutable boost::mt19937 mGen;
};
