#pragma once

#include <string>
#include <map>
#include <typeinfo>
#include <boost/any.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "SparseMatrixBuffer.h"
#include "Tensor3Buffer.h"

#define BufferCollectionKey_t std::string

class BufferCollection
{
public:
    BufferCollection();

#define DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(BUFFER_TYPE) \
bool Has ## BUFFER_TYPE(BufferCollectionKey_t name) const; \
void Add ## BUFFER_TYPE(BufferCollectionKey_t name, BUFFER_TYPE const& data ); \
void Append ## BUFFER_TYPE(BufferCollectionKey_t name, BUFFER_TYPE const& data ); \
const BUFFER_TYPE& Get ## BUFFER_TYPE(const BufferCollectionKey_t& name) const; \
BUFFER_TYPE& Get ## BUFFER_TYPE(const BufferCollectionKey_t& name);

    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32VectorBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64VectorBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32VectorBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64VectorBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32MatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64MatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32MatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64MatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32SparseMatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64SparseMatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32SparseMatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64SparseMatrixBuffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float32Tensor3Buffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Float64Tensor3Buffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int32Tensor3Buffer)
    DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE(Int64Tensor3Buffer)


#undef DECLARE_BUFFER_SWIG_INTERFACE_FOR_TYPE

    bool HasBuffer(BufferCollectionKey_t name) const;

    template<typename BufferType>
    void AddBuffer(BufferCollectionKey_t name, BufferType const& data);
    template<typename BufferType>
    BufferType& GetOrAddBuffer(BufferCollectionKey_t name);
    template<typename BufferType>
    void AppendBuffer(BufferCollectionKey_t name, BufferType const& buffer);
    template<typename BufferType>
    BufferType const& GetBuffer(BufferCollectionKey_t name) const;
    template<typename BufferType>
    BufferType& GetBuffer(BufferCollectionKey_t name);
    template<typename BufferType>
    BufferType const* GetBufferPtr(BufferCollectionKey_t name) const;
    template<typename BufferType>
    BufferType* GetBufferPtr(BufferCollectionKey_t name);

// private:
    // Checks for a buffer of a specific type.
    //
    // Remove this when the transition to the templated interface is complete.
    // Right now it's being used to check for violated assumptions during the
    // transition.
    template<typename BufferType>
    bool HasBuffer(BufferCollectionKey_t name) const {
        if (HasBuffer(name)) {
            BufferMapType::const_iterator bufferIter = mBuffers.find(name);
            return bufferIter->second.type() == typeid(BufferType);
        }
        return false;
    }

private:
    typedef std::map<BufferCollectionKey_t, boost::any> BufferMapType;

    BufferMapType mBuffers;
};


template<typename BufferType>
void BufferCollection::AddBuffer(BufferCollectionKey_t name, BufferType const& buffer)
{
    mBuffers[name] = boost::any(buffer);
}

template<typename BufferType>
BufferType& BufferCollection::GetOrAddBuffer(BufferCollectionKey_t name)
{
    if( !HasBuffer<BufferType>(name) )
    {
        mBuffers[name] = boost::any(BufferType());
    }
    return GetBuffer<BufferType>(name);
}

template<typename BufferType>
void BufferCollection::AppendBuffer(BufferCollectionKey_t name, BufferType const& buffer)
{
    if (!HasBuffer(name)) {
        AddBuffer(name, buffer);
    }
    else {
        GetBuffer<BufferType>(name).Append(buffer);
    }
}

template<typename BufferType>
BufferType const& BufferCollection::GetBuffer(BufferCollectionKey_t name) const
{
    BufferType const* ptr = GetBufferPtr<BufferType>(name);
    return *ptr;
}

template<typename BufferType>
BufferType& BufferCollection::GetBuffer(BufferCollectionKey_t name)
{
    BufferType* ptr = GetBufferPtr<BufferType>(name);
    return *ptr;
}

template<typename BufferType>
BufferType const* BufferCollection::GetBufferPtr(BufferCollectionKey_t name) const
{
    ASSERT(HasBuffer(name));
    BufferMapType::const_iterator bufferIter = mBuffers.find(name);
    // use pointers so any_cast doesn't copy the buffer
    return boost::any_cast<BufferType>(&bufferIter->second);
}

template<typename BufferType>
BufferType* BufferCollection::GetBufferPtr(BufferCollectionKey_t name)
{
    ASSERT(HasBuffer(name));
    BufferMapType::iterator bufferIter = mBuffers.find(name);
    // use pointers so any_cast doesn't copy the buffer
    return boost::any_cast<BufferType>(&bufferIter->second);
}


//Using #define for compatibility with swig
#define X_FLOAT_DATA    "x_float"
#define SAMPLE_WEIGHTS  "SampleWeights"
#define CLASS_LABELS    "y_class"
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