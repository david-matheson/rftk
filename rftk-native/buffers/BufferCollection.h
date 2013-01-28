#pragma once

#include <string>
#include <map>

#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"

//Using #define for compatibility with swig
#define X_FLOAT_DATA    "X_Float"
#define SAMPLE_WEIGHTS  "SampleWeights"
#define CLASS_LABELS    "ClassLabels"
#define FEATURE_VALUES  "Feature_Values"
#define HISTOGRAM_LEFT  "histogram_Left"
#define HISTOGRAM_RIGHT "histogram_Right"
#define THRESHOLDS      "Thresholds"


class BufferCollection
{
public:
    bool HasFloat32MatrixBuffer(std::string name) const;
    void AddFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data );
    void AppendVerticalFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data );
    const Float32MatrixBuffer& GetFloat32MatrixBuffer(const std::string& name) const;
    Float32MatrixBuffer& GetFloat32MatrixBuffer(const std::string& name);

    bool HasInt32MatrixBuffer(std::string name) const;
    void AddInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data );
    void AppendVerticalInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data );
    const Int32MatrixBuffer& GetInt32MatrixBuffer(const std::string& name) const;
    Int32MatrixBuffer& GetInt32MatrixBuffer(const std::string& name);

    bool HasFloat32Tensor3Buffer(std::string name) const;
    void AddFloat32Tensor3Buffer(std::string name, const Float32Tensor3Buffer& data );
    const Float32Tensor3Buffer& GetFloat32Tensor3Buffer(std::string name) const;
    Float32Tensor3Buffer& GetFloat32Tensor3Buffer(std::string name);

    bool HasInt32Tensor3Buffer(std::string name) const;
    void AddInt32Tensor3Buffer(std::string name, const Int32Tensor3Buffer& data ) ;
    const Int32Tensor3Buffer& GetInt32Tensor3Buffer(std::string name) const;
    Int32Tensor3Buffer& GetInt32Tensor3Buffer(std::string name);

private:
    std::map<std::string, Float32MatrixBuffer> mFloatMatrixBuffers;
    std::map<std::string, Int32MatrixBuffer> mIntMatrixBuffers;
    std::map<std::string, Float32Tensor3Buffer> mFloatImgBuffers;
    std::map<std::string, Int32Tensor3Buffer> mIntImgBuffers;
};