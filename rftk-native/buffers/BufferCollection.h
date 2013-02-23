#pragma once

#include <string>
#include <map>
#include <boost/any.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"

//Using #define for compatibility with swig
#define X_FLOAT_DATA    "X_Float"
#define SAMPLE_WEIGHTS  "SampleWeights"
#define CLASS_LABELS    "ClassLabels"
#define FEATURE_VALUES  "Feature_Values"
#define HISTOGRAM_LEFT            "Histogram_Left"
#define HISTOGRAM_RIGHT           "Histogram_Right"
#define THRESHOLDS                "Thresholds"
#define THRESHOLD_COUNTS          "ThresholdCounts"
#define IMPURITY_HISTOGRAM_LEFT   "Impurity_Histogram_Left"
#define IMPURITY_HISTOGRAM_RIGHT  "Impurity_Histogram_Right"
#define YS_HISTOGRAM_LEFT         "Ys_Histogram_Left"
#define YS_HISTOGRAM_RIGHT        "Ys_Histogram_Right"

#define PIXEL_INDICES   "PixelIndices"
#define DEPTH_IMAGES    "DepthImages"
#define OFFSET_SCALES   "OffsetScales"

namespace detail {

// Modelled after boost::any, but avoids copying.
//
// From the boost::any documentation:
//
// Values are strongly informational objects for which identity is not
// significant, i.e. the focus is principally on their state content and any
// behavior organized around that.
//
// The focus of this class is exactly the opposite.  In this class identity of
// the value is important, and care is taken to ensure that identity is
// preserved.
//
// In particular this means that if you put an AnyBuffer into an stl collection
// and then pull it out again the buffer it contains will be the same _instance_
// as the buffer you put in, even if the AnyBuffer container was copied
// internally by the collection.
//
// Identity is changed by assignment.  Assigning one A = B causes A to take on
// the identity of B.



// used internally by AnyBuffer
class BufferHolder
{
public:
    virtual ~BufferHolder() {}
    virtual BufferHolder* clone() const = 0;
};

template<typename ValueType>
class AnyBufferHolder : public BufferHolder
{
public:
    AnyBufferHolder(ValueType* buffer)
        : mBuffer(buffer)
    {
    }

    virtual BufferHolder* clone() const {
        return new AnyBufferHolder<ValueType>(mBuffer);
    }

public:
    ValueType *mBuffer;

private:
    // unimplemented: prevent the compiler from generating a default assignment
    // operator for this class.
    AnyBufferHolder& operator=(AnyBufferHolder const&);
};

class AnyBuffer {
public:
    AnyBuffer()
        : mHolder(0)
    {
    }

    AnyBuffer(AnyBuffer const& other)
        : mHolder(other.mHolder ? other.mHolder->clone() : 0)
    {
    }

    template<typename ValueType>
    AnyBuffer(ValueType& buffer)
        : mHolder(new AnyBufferHolder<ValueType>(&buffer))
    {
    }

    ~AnyBuffer() { delete mHolder; }

    template<typename ValueType>
    ValueType& GetBuffer()
    {
        return *dynamic_cast<AnyBufferHolder<ValueType> *>(mHolder)->mBuffer;
    }

    template<typename ValueType>
    ValueType const& GetBuffer() const
    {
        return *dynamic_cast<AnyBufferHolder<ValueType> const*>(mHolder)->mBuffer;
    }

    AnyBuffer& operator=(AnyBuffer const& other)
    {
        mHolder = other.mHolder ? other.mHolder->clone() : 0;
        return *this;
    }

private:
    BufferHolder* mHolder;
};

} // namespace detail

class BufferCollection
{
public:
    bool HasFloat32VectorBuffer(std::string name) const;
    void AddFloat32VectorBuffer(std::string name, const Float32VectorBuffer& data );
    void AppendFloat32VectorBuffer(std::string name, const Float32VectorBuffer& data );
    const Float32VectorBuffer& GetFloat32VectorBuffer(const std::string& name) const;
    Float32VectorBuffer& GetFloat32VectorBuffer(const std::string& name);

    bool HasInt32VectorBuffer(std::string name) const;
    void AddInt32VectorBuffer(std::string name, const Int32VectorBuffer& data );
    void AppendInt32VectorBuffer(std::string name, const Int32VectorBuffer& data );
    const Int32VectorBuffer& GetInt32VectorBuffer(const std::string& name) const;
    Int32VectorBuffer& GetInt32VectorBuffer(const std::string& name);

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

    bool HasBuffer(std::string name) const;

    template<typename BufferType>
    void AddBuffer(std::string name, BufferType& data);
    template<typename BufferType>
    BufferType const& GetBuffer(std::string name) const;
    template<typename BufferType>
    BufferType& GetBuffer(std::string name);

private:

    typedef std::map<std::string, detail::AnyBuffer> BufferMapType;

    std::map<std::string, Float32VectorBuffer> mFloat32VectorBuffers;
    std::map<std::string, Int32VectorBuffer> mInt32VectorBuffers;
    std::map<std::string, Float32MatrixBuffer> mFloat32MatrixBuffers;
    std::map<std::string, Int32MatrixBuffer> mInt32MatrixBuffers;
    std::map<std::string, Float32Tensor3Buffer> mFloat32Tensor3Buffers;
    std::map<std::string, Int32Tensor3Buffer> mInt32Tensor3Buffers;

    BufferMapType mBuffers;
};


template<typename BufferType>
void BufferCollection::AddBuffer(std::string name, BufferType& buffer)
{
    mBuffers[name] = detail::AnyBuffer(buffer);
}

template<typename BufferType>
BufferType const& BufferCollection::GetBuffer(std::string name) const
{
    ASSERT(HasBuffer(name));
    BufferMapType::const_iterator bufferIter = mBuffers.find(name);
    return bufferIter->second.GetBuffer<BufferType>();
}

template<typename BufferType>
BufferType& BufferCollection::GetBuffer(std::string name)
{
    ASSERT(HasBuffer(name));
    BufferMapType::iterator bufferIter = mBuffers.find(name);
    return bufferIter->second.GetBuffer<BufferType>();
}
