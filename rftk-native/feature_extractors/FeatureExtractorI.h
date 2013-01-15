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
                        MatrixBufferFloat& featureValuesOUT )  const// #tests X #samples
                        {}

    virtual MatrixBufferFloat CreateFloatParams() const { return MatrixBufferFloat(); }
    virtual MatrixBufferInt CreateIntParams() const { return MatrixBufferInt(); }

    virtual int GetFloatParamsDim() const { return 1; }
    virtual int GetIntParamsDim() const { return 1; }

    // virtual void UpdateStateFromForest(const Forest& forest) { }

    virtual int GetUID() const { return 0; }

};