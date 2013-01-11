#pragma once

class MatrixBufferInt;
class MatrixBufferFloat;
class BufferCollection;

class FeatureExtractorI {
public:
    virtual void Extract(const BufferCollection& data,
                        const MatrixBufferInt& sampleIndices,
                        const MatrixBufferInt& intFeatureParams,
                        const MatrixBufferFloat& floatFeatureParams,
                        MatrixBufferFloat& featureValuesOUT ) // #tests X #samples
                        {}


    // virtual void UpdateStateFromForest(const Forest& forest) { }

    virtual int GetUID() { return 0; }

};