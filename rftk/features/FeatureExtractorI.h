#pragma once

class MatrixBufferInt;
class MatrixBufferFloat;

class FeatureExtractorI {
public:
    virtual void Extract(const MatrixBufferInt& sampleIndices, 
                        const MatrixBufferInt& intFeatureParams,
                        const MatrixBufferFloat& floatFeatureParams,
                        MatrixBufferFloat& featureValuesOUT ) // #tests X #samples
                        {}


    // virtual void UpdateStateFromForest(const Forest& forest) { }

    virtual int GetUID() { return 0; }

};