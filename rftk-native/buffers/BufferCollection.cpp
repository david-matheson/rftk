#include "assert_util.h"
#include "BufferCollection.h"

BufferCollection::BufferCollection()
    : mBuffers()
{
}

// HasBUFFER_TYPE function makes sure that if there is a buffer with the name
// you're asking for then it is of the type you are requesting.  The old
// interface would allow two buffers with the same name but different types ot
// co-exist.  I don't think we were using that feature, but hopefully this check
// will catch any instances where we are (if they exist).
#define DEFINE_BUFFER_LEGACY_INTERFACE_FOR_TYPE(BUFFER_TYPE) \
bool BufferCollection::Has ## BUFFER_TYPE(std::string name) const \
{ \
    ASSERT(!(HasBuffer(name) && !HasBuffer<BUFFER_TYPE>(name))); \
    return HasBuffer(name); \
} \
void BufferCollection::Add ## BUFFER_TYPE(std::string name, BUFFER_TYPE const& data ) \
{ \
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

DEFINE_BUFFER_LEGACY_INTERFACE_FOR_TYPE(Float32VectorBuffer)
DEFINE_BUFFER_LEGACY_INTERFACE_FOR_TYPE(Int32VectorBuffer)
DEFINE_BUFFER_LEGACY_INTERFACE_FOR_TYPE(Float32MatrixBuffer)
DEFINE_BUFFER_LEGACY_INTERFACE_FOR_TYPE(Int32MatrixBuffer)
DEFINE_BUFFER_LEGACY_INTERFACE_FOR_TYPE(Float32Tensor3Buffer)
DEFINE_BUFFER_LEGACY_INTERFACE_FOR_TYPE(Int32Tensor3Buffer)

bool BufferCollection::HasBuffer(std::string name) const
{
    return (mBuffers.find(name) != mBuffers.end());
}

