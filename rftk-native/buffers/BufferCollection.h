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
    bool HasMatrixBufferFloat(std::string name) const;
    void AddMatrixBufferFloat(std::string name, const MatrixBufferFloat& data );
    void AppendVerticalMatrixBufferFloat(std::string name, const MatrixBufferFloat& data );
    const MatrixBufferFloat& GetMatrixBufferFloat(const std::string& name) const;

    bool HasMatrixBufferInt(std::string name) const;
    void AddMatrixBufferInt(std::string name, const MatrixBufferInt& data );
    void AppendVerticalMatrixBufferInt(std::string name, const MatrixBufferInt& data );
    const MatrixBufferInt& GetMatrixBufferInt(const std::string& name) const;

    bool HasImgBufferFloat(std::string name) const;
    void AddImgBufferFloat(std::string name, const ImgBufferFloat& data );
    ImgBufferFloat GetImgBufferFloat(std::string name);

    bool HasImgBufferInt(std::string name) const;
    void AddImgBufferInt(std::string name, const ImgBufferInt& data ) ;
    ImgBufferInt GetImgBufferInt(std::string name);

private:
    std::map<std::string, MatrixBufferFloat> mFloatMatrixBuffers;
    std::map<std::string, MatrixBufferInt> mIntMatrixBuffers;
    std::map<std::string, ImgBufferFloat> mFloatImgBuffers;
    std::map<std::string, ImgBufferInt> mIntImgBuffers;
};