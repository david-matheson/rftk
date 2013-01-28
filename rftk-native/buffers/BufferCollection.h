#pragma once

#include <string>
#include <map>

#include "MatrixBuffer.h"
#include "ImgBuffer.h"

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

    bool HasImgBufferFloat(std::string name) const;
    void AddImgBufferFloat(std::string name, const ImgBufferFloat& data );
    const ImgBufferFloat& GetImgBufferFloat(std::string name) const;
    ImgBufferFloat& GetImgBufferFloat(std::string name);

    bool HasImgBufferInt(std::string name) const;
    void AddImgBufferInt(std::string name, const ImgBufferInt& data ) ;
    const ImgBufferInt& GetImgBufferInt(std::string name) const;
    ImgBufferInt& GetImgBufferInt(std::string name);

private:
    std::map<std::string, Float32MatrixBuffer> mFloatMatrixBuffers;
    std::map<std::string, Int32MatrixBuffer> mIntMatrixBuffers;
    std::map<std::string, ImgBufferFloat> mFloatImgBuffers;
    std::map<std::string, ImgBufferInt> mIntImgBuffers;
};