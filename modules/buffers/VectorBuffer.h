#pragma once

#include <vector>
#include <limits>
#include <iostream>

#include "assert_util.h"

template <class T>
class VectorBufferTemplate {
public:
    VectorBufferTemplate();
    VectorBufferTemplate(int n);
    VectorBufferTemplate(float* data, int n);
    VectorBufferTemplate(double* data, int n);
    VectorBufferTemplate(int* data, int n);
    VectorBufferTemplate(long long* data, int n);
    ~VectorBufferTemplate();

    void Resize(int n);
    void Zero();
    void SetAll(const T value);

    int GetN() const { return mN; }

    void Set(int n, T value);
    T Get(int n) const;
    void SetUnsafe(int n, T value);
    T GetUnsafe(int n) const;

    void Incr(int n, T value);

    T GetMax() const;
    T GetMin() const;

    void Append(const VectorBufferTemplate<T>& buffer);
    VectorBufferTemplate<T> Slice(const VectorBufferTemplate<int>& indices) const;

    void AsNumpy1dFloat32(float* outfloat1d, int n) const;
    void AsNumpy1dInt32(int* outint1d, int n) const;


    void Print() const;

private:
    std::vector< T > mData;
    int mN;
};

template <class T>
VectorBufferTemplate<T>::VectorBufferTemplate()
: mData()
, mN(0)
{
}

template <class T>
VectorBufferTemplate<T>::VectorBufferTemplate(int n)
: mData( n )
, mN(n)
{
}

template <class T>
VectorBufferTemplate<T>::VectorBufferTemplate(float* data, int n)
: mData( data, data + n )
, mN(n)
{
    for(int i=0; i<n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
VectorBufferTemplate<T>::VectorBufferTemplate(double* data, int n)
: mData( n )
, mN(n)
{
    for(int i=0; i<n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
VectorBufferTemplate<T>::VectorBufferTemplate(int* data, int n)
: mData( n )
, mN(n)
{
    for(int i=0; i<n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
VectorBufferTemplate<T>::VectorBufferTemplate(long long* data, int n)
: mData( n )
, mN(n)
{
    for(int i=0; i<n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
VectorBufferTemplate<T>::~VectorBufferTemplate()
{
}

template <class T>
void VectorBufferTemplate<T>::Resize( int n)
{
    if(static_cast<size_t>(n) > mData.size())
    {
        mData.resize(n);
    }
    mN = n;
}

template <class T>
void VectorBufferTemplate<T>::Zero()
{
    SetAll(static_cast<T>(0));
}

template <class T>
void VectorBufferTemplate<T>::SetAll(const T value)
{
    std::fill(mData.begin(), mData.end(), value);
}

template <class T>
void VectorBufferTemplate<T>::Set(int n, T value)
{
    ASSERT_VALID_RANGE(n, 0, mN)
    mData[n] = value;
}

template <class T>
T VectorBufferTemplate<T>::Get(int n) const
{
    ASSERT_VALID_RANGE(n, 0, mN)
    return mData[n];
}

template <class T>
void VectorBufferTemplate<T>::SetUnsafe(int n, T value)
{
    mData[n] = value;
}

template <class T>
T VectorBufferTemplate<T>::GetUnsafe(int n) const
{
    return mData[n];
}

template <class T>
void VectorBufferTemplate<T>::Incr(int n, T value)
{
    mData[n] += value;
}

template <class T>
T VectorBufferTemplate<T>::GetMax() const
{
    T max = std::numeric_limits<T>::min();
    for(int i=0; i<mN; i++)
    {
        max = (max > mData[i]) ? max : mData[i];
    }
    return max;
}

template <class T>
T VectorBufferTemplate<T>::GetMin() const
{
    T min = std::numeric_limits<T>::max();
    for(int i=0; i<mN; i++)
    {
        min = (min < mData[i]) ? min : mData[i];
    }
    return min;
}

template <class T>
void VectorBufferTemplate<T>::Append(const VectorBufferTemplate<T>& buffer)
{
    const int oldN = mN;
    Resize(mN + buffer.GetN());
    for(int n=0; n<buffer.GetN(); n++)
    {
        Set(n+oldN, buffer.Get(n));
    }
}

template <class T>
VectorBufferTemplate<T> VectorBufferTemplate<T>::Slice(const VectorBufferTemplate<int>& indices) const
{
    VectorBufferTemplate<T> sliced(indices.GetN());
    for(int i=0; i<indices.GetN(); i++)
    {
        int r = indices.Get(i);
        sliced.Set(i, Get(r));
    }
    return sliced;
}

template <class T>
void VectorBufferTemplate<T>::AsNumpy1dFloat32(float* outfloat1d, int n) const
{
    ASSERT_ARG_DIM_1D(n, mN)
    for(int i=0; i<std::min(n,mN); i++)
    {
        outfloat1d[i] = static_cast<float>(mData[i]);
    }
}

template <class T>
void VectorBufferTemplate<T>::AsNumpy1dInt32(int* outint1d, int n) const
{
    ASSERT_ARG_DIM_1D(n, mN)
    for(int i=0; i<std::min(n,mN); i++)
    {
        outint1d[i] = static_cast<int>(mData[i]);
    }
}

template <class T>
void VectorBufferTemplate<T>::Print() const
{
    std::cout << "[" << mN << "]" << std::endl;
    std::cout << "[";
    for(int n=0; n<mN; n++)
    {
        std::cout << Get(n) << " ";
    }
    std::cout << "]" << std::endl;
}

typedef VectorBufferTemplate<float> Float32VectorBuffer;
typedef VectorBufferTemplate<double> Float64VectorBuffer;
typedef VectorBufferTemplate<int> Int32VectorBuffer;
typedef VectorBufferTemplate<long long> Int64VectorBuffer;

Float32VectorBuffer Float32Vector(float* float1d, int n);
Float64VectorBuffer Float64Vector(double* double1d, int n);
Int32VectorBuffer Int32Vector(int* int1d, int n);
Int64VectorBuffer Int64Vector(long long* long1d, int n);




