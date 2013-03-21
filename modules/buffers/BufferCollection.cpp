#include "asserts/asserts.h"
#include "BufferCollection.h"

BufferCollection::BufferCollection()
    : mBuffers()
{
}

#define DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(BUFFER_TYPE) \
bool BufferCollection::Has ## BUFFER_TYPE(std::string name) const \
{ \
    return HasBuffer(name) && HasBuffer<BUFFER_TYPE>(name); \
} \
void BufferCollection::Add ## BUFFER_TYPE(std::string name, BUFFER_TYPE const& data ) \
{ \
    ASSERT(!(HasBuffer(name) && !HasBuffer<BUFFER_TYPE>(name))); \
    AddBuffer<BUFFER_TYPE>(name, data); \
} \
void BufferCollection::Append ## BUFFER_TYPE(std::string name, BUFFER_TYPE const& data ) \
{ \
    AppendBuffer<BUFFER_TYPE>(name, data); \
} \
const BUFFER_TYPE& BufferCollection::Get ## BUFFER_TYPE(const std::string& name) const \
{ \
    return GetBuffer<BUFFER_TYPE>(name); \
} \
BUFFER_TYPE& BufferCollection::Get ## BUFFER_TYPE(const std::string& name) \
{ \
    return GetBuffer<BUFFER_TYPE>(name); \
}

DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32VectorBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64VectorBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32VectorBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64VectorBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32MatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64MatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32MatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64MatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32SparseMatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64SparseMatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32SparseMatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64SparseMatrixBuffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32Tensor3Buffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64Tensor3Buffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32Tensor3Buffer)
DEFINE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64Tensor3Buffer)

bool BufferCollection::HasBuffer(std::string name) const
{
    return (mBuffers.find(name) != mBuffers.end());
}

