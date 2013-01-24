#include "BufferCollection.h"

bool BufferCollection::HasMatrixBufferFloat(std::string name) const
{
    return (mFloatMatrixBuffers.find(name) != mFloatMatrixBuffers.end());
}

void BufferCollection::AddMatrixBufferFloat(std::string name, const MatrixBufferFloat& data )
{
    mFloatMatrixBuffers[name] = data;
}

void BufferCollection::AppendVerticalMatrixBufferFloat(std::string name, const MatrixBufferFloat& data )
{
    if( !HasMatrixBufferFloat(name) )
    {
        mFloatMatrixBuffers[name] = data;
    }
    else
    {
        mFloatMatrixBuffers[name].AppendVertical(data);
    }
}

const MatrixBufferFloat& BufferCollection::GetMatrixBufferFloat(const std::string& name) const
{
    return mFloatMatrixBuffers.find(name)->second;
}

bool BufferCollection::HasMatrixBufferInt(std::string name) const
{
    return (mIntMatrixBuffers.find(name) != mIntMatrixBuffers.end());
}

void BufferCollection::AddMatrixBufferInt(std::string name, const MatrixBufferInt& data )
{
    mIntMatrixBuffers[name] = data;
}

void BufferCollection::AppendVerticalMatrixBufferInt(std::string name, const MatrixBufferInt& data )
{
    if( !HasMatrixBufferInt(name) )
    {
        mIntMatrixBuffers[name] = data;
    }
    else
    {
        mIntMatrixBuffers[name].AppendVertical(data);
    }
}

const MatrixBufferInt& BufferCollection::GetMatrixBufferInt(const std::string& name) const
{
    return mIntMatrixBuffers.find(name)->second;
}


bool BufferCollection::HasImgBufferFloat(std::string name) const
{
    return (mFloatImgBuffers.find(name) != mFloatImgBuffers.end());
}

void BufferCollection::AddImgBufferFloat(std::string name, const ImgBufferFloat& data )
{
    mFloatImgBuffers[name] = data;
}

ImgBufferFloat BufferCollection::GetImgBufferFloat(std::string name)
{
    return mFloatImgBuffers[name];
}

bool BufferCollection::HasImgBufferInt(std::string name) const
{
    return (mIntImgBuffers.find(name) != mIntImgBuffers.end());
}

void BufferCollection::AddImgBufferInt(std::string name, const ImgBufferInt& data )
{
    mIntImgBuffers[name] = data;
}

ImgBufferInt BufferCollection::GetImgBufferInt(std::string name)
{
    return mIntImgBuffers[name];
}