#include "BufferCollection.h"

bool BufferCollection::HasFloat32MatrixBuffer(std::string name) const
{
    return (mFloatMatrixBuffers.find(name) != mFloatMatrixBuffers.end());
}

void BufferCollection::AddFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data )
{
    mFloatMatrixBuffers[name] = data;
}

void BufferCollection::AppendVerticalFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data )
{
    if( !HasFloat32MatrixBuffer(name) )
    {
        mFloatMatrixBuffers[name] = data;
    }
    else
    {
        mFloatMatrixBuffers[name].AppendVertical(data);
    }
}

const Float32MatrixBuffer& BufferCollection::GetFloat32MatrixBuffer(const std::string& name) const
{
    return mFloatMatrixBuffers.find(name)->second;
}

Float32MatrixBuffer& BufferCollection::GetFloat32MatrixBuffer(const std::string& name)
{
    return mFloatMatrixBuffers.find(name)->second;
}

bool BufferCollection::HasInt32MatrixBuffer(std::string name) const
{
    return (mIntMatrixBuffers.find(name) != mIntMatrixBuffers.end());
}

void BufferCollection::AddInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data )
{
    mIntMatrixBuffers[name] = data;
}

void BufferCollection::AppendVerticalInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data )
{
    if( !HasInt32MatrixBuffer(name) )
    {
        mIntMatrixBuffers[name] = data;
    }
    else
    {
        mIntMatrixBuffers[name].AppendVertical(data);
    }
}

const Int32MatrixBuffer& BufferCollection::GetInt32MatrixBuffer(const std::string& name) const
{
    return mIntMatrixBuffers.find(name)->second;
}

Int32MatrixBuffer& BufferCollection::GetInt32MatrixBuffer(const std::string& name)
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

const ImgBufferFloat& BufferCollection::GetImgBufferFloat(std::string name) const
{
    return mFloatImgBuffers.find(name)->second;
}

ImgBufferFloat& BufferCollection::GetImgBufferFloat(std::string name)
{
    return mFloatImgBuffers.find(name)->second;
}

bool BufferCollection::HasImgBufferInt(std::string name) const
{
    return (mIntImgBuffers.find(name) != mIntImgBuffers.end());
}

void BufferCollection::AddImgBufferInt(std::string name, const ImgBufferInt& data )
{
    mIntImgBuffers[name] = data;
}

const ImgBufferInt& BufferCollection::GetImgBufferInt(std::string name) const
{
    return mIntImgBuffers.find(name)->second;
}

ImgBufferInt& BufferCollection::GetImgBufferInt(std::string name)
{
    return mIntImgBuffers.find(name)->second;
}