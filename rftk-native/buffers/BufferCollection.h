#pragma once

#include <string>
#include <map>

#include "MatrixBuffer.h"
#include "ImgBuffer.h"

class BufferCollection
{
public:
    bool HasMatrixBufferFloat(std::string name);
    void AddMatrixBufferFloat(std::string name, const MatrixBufferFloat& data );
    MatrixBufferFloat GetMatrixBufferFloat(std::string name);

    bool HasMatrixBufferInt(std::string name);
    void AddMatrixBufferInt(std::string name, const MatrixBufferInt& data );
    MatrixBufferInt GetMatrixBufferInt(std::string name);

    bool HasImgBufferFloat(std::string name);
    void AddImgBufferFloat(std::string name, const ImgBufferFloat& data );
    ImgBufferFloat GetImgBufferFloat(std::string name);

    bool HasImgBufferInt(std::string name);
    void AddImgBufferInt(std::string name, const ImgBufferInt& data );
    ImgBufferInt GetImgBufferInt(std::string name);

private:
    std::map<std::string, MatrixBufferFloat> mFloatMatrixBuffers;
    std::map<std::string, MatrixBufferInt> mIntMatrixBuffers;
    std::map<std::string, ImgBufferFloat> mFloatImgBuffers;
    std::map<std::string, ImgBufferInt> mIntImgBuffers;
};