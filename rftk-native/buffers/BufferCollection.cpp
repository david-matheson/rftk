#include "assert_util.h"
#include "BufferCollection.h"

bool BufferCollection::HasFloat32VectorBuffer(std::string name) const
{
    return (mFloat32VectorBuffers.find(name) != mFloat32VectorBuffers.end());
}

void BufferCollection::AddFloat32VectorBuffer(std::string name, const Float32VectorBuffer& data )
{
    mFloat32VectorBuffers[name] = data;
}

void BufferCollection::AppendFloat32VectorBuffer(std::string name, const Float32VectorBuffer& data )
{
    if( !HasFloat32VectorBuffer(name) )
    {
        mFloat32VectorBuffers[name] = data;
    }
    else
    {
        mFloat32VectorBuffers[name].Append(data);
    }
}

const Float32VectorBuffer& BufferCollection::GetFloat32VectorBuffer(const std::string& name) const
{
    ASSERT(HasFloat32VectorBuffer(name))
    return mFloat32VectorBuffers.find(name)->second;
}

Float32VectorBuffer& BufferCollection::GetFloat32VectorBuffer(const std::string& name)
{
    ASSERT(HasFloat32VectorBuffer(name))
    return mFloat32VectorBuffers.find(name)->second;
}


bool BufferCollection::HasInt32VectorBuffer(std::string name) const
{
    return (mInt32VectorBuffers.find(name) != mInt32VectorBuffers.end());
}

void BufferCollection::AddInt32VectorBuffer(std::string name, const Int32VectorBuffer& data )
{
    mInt32VectorBuffers[name] = data;
}

void BufferCollection::AppendInt32VectorBuffer(std::string name, const Int32VectorBuffer& data )
{
    if( !HasInt32VectorBuffer(name) )
    {
        mInt32VectorBuffers[name] = data;
    }
    else
    {
        mInt32VectorBuffers[name].Append(data);
    }
}

const Int32VectorBuffer& BufferCollection::GetInt32VectorBuffer(const std::string& name) const
{
    ASSERT(HasInt32VectorBuffer(name))
    return mInt32VectorBuffers.find(name)->second;
}

Int32VectorBuffer& BufferCollection::GetInt32VectorBuffer(const std::string& name)
{
    ASSERT(HasInt32VectorBuffer(name))
    return mInt32VectorBuffers.find(name)->second;
}

bool BufferCollection::HasFloat32MatrixBuffer(std::string name) const
{
    return (mFloat32MatrixBuffers.find(name) != mFloat32MatrixBuffers.end());
}

void BufferCollection::AddFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data )
{
    mFloat32MatrixBuffers[name] = data;
}

void BufferCollection::AppendVerticalFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data )
{
    if( !HasFloat32MatrixBuffer(name) )
    {
        mFloat32MatrixBuffers[name] = data;
    }
    else
    {
        mFloat32MatrixBuffers[name].AppendVertical(data);
    }
}

const Float32MatrixBuffer& BufferCollection::GetFloat32MatrixBuffer(const std::string& name) const
{
    ASSERT(HasFloat32MatrixBuffer(name))
    return mFloat32MatrixBuffers.find(name)->second;
}

Float32MatrixBuffer& BufferCollection::GetFloat32MatrixBuffer(const std::string& name)
{
    ASSERT(HasFloat32MatrixBuffer(name))
    return mFloat32MatrixBuffers.find(name)->second;
}

bool BufferCollection::HasInt32MatrixBuffer(std::string name) const
{
    return (mInt32MatrixBuffers.find(name) != mInt32MatrixBuffers.end());
}

void BufferCollection::AddInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data )
{
    mInt32MatrixBuffers[name] = data;
}

void BufferCollection::AppendVerticalInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data )
{
    if( !HasInt32MatrixBuffer(name) )
    {
        mInt32MatrixBuffers[name] = data;
    }
    else
    {
        mInt32MatrixBuffers[name].AppendVertical(data);
    }
}

const Int32MatrixBuffer& BufferCollection::GetInt32MatrixBuffer(const std::string& name) const
{
    ASSERT(HasInt32MatrixBuffer(name))
    return mInt32MatrixBuffers.find(name)->second;
}

Int32MatrixBuffer& BufferCollection::GetInt32MatrixBuffer(const std::string& name)
{
    ASSERT(HasInt32MatrixBuffer(name))
    return mInt32MatrixBuffers.find(name)->second;
}

bool BufferCollection::HasFloat32Tensor3Buffer(std::string name) const
{
    return (mFloat32Tensor3Buffers.find(name) != mFloat32Tensor3Buffers.end());
}

void BufferCollection::AddFloat32Tensor3Buffer(std::string name, const Float32Tensor3Buffer& data )
{
    mFloat32Tensor3Buffers[name] = data;
}

const Float32Tensor3Buffer& BufferCollection::GetFloat32Tensor3Buffer(std::string name) const
{
    ASSERT(HasFloat32Tensor3Buffer(name))
    return mFloat32Tensor3Buffers.find(name)->second;
}

Float32Tensor3Buffer& BufferCollection::GetFloat32Tensor3Buffer(std::string name)
{
    ASSERT(HasFloat32Tensor3Buffer(name))
    return mFloat32Tensor3Buffers.find(name)->second;
}

bool BufferCollection::HasInt32Tensor3Buffer(std::string name) const
{
    return (mInt32Tensor3Buffers.find(name) != mInt32Tensor3Buffers.end());
}

void BufferCollection::AddInt32Tensor3Buffer(std::string name, const Int32Tensor3Buffer& data )
{
    mInt32Tensor3Buffers[name] = data;
}

const Int32Tensor3Buffer& BufferCollection::GetInt32Tensor3Buffer(std::string name) const
{
    ASSERT(HasInt32Tensor3Buffer(name))
    return mInt32Tensor3Buffers.find(name)->second;
}

Int32Tensor3Buffer& BufferCollection::GetInt32Tensor3Buffer(std::string name)
{
    ASSERT(HasInt32Tensor3Buffer(name))
    return mInt32Tensor3Buffers.find(name)->second;
}

bool BufferCollection::HasBuffer(std::string name) const
{
    return (mBuffers.find(name) != mBuffers.end());
}

