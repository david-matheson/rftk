#pragma once

#include <vector>
#include <limits>
#include <iostream>

#include <asserts.h>
#include "VectorBuffer.h"

template <class T>
class MatrixBufferTemplate {
public:
    MatrixBufferTemplate();
    MatrixBufferTemplate(int m, int n);
    MatrixBufferTemplate(int m, int n, T value);
    MatrixBufferTemplate(float* data, int m, int n);
    MatrixBufferTemplate(double* data, int m, int n);
    MatrixBufferTemplate(int* data, int m, int n);
    MatrixBufferTemplate(long long* data, int m, int n);
    ~MatrixBufferTemplate();

    void Resize(int m, int n);
    void Resize(int m, int n, T value);
    void Zero();
    void SetAll(const T value);

    int GetM() const { return mM; }
    int GetN() const { return mN; }

    void Set(int m, int n, T value);
    T Get(int m, int n) const;
    void SetUnsafe(int m, int n, T value);
    T GetUnsafe(int m, int n) const;
    void Incr(int m, int n, T value);

    const T* GetRowPtrUnsafe(int m) const;

    T GetMax() const;
    T GetMin() const;

    T SumRow(int m) const;
    void NormalizeRow(int m);

    void Append(const MatrixBufferTemplate<T>& buffer);
    MatrixBufferTemplate<T> Transpose() const;
    MatrixBufferTemplate<T> Slice(const VectorBufferTemplate<int>& indices) const;
    MatrixBufferTemplate<T> SliceRow(const int row) const;

    void AsNumpy2dFloat32(float* outfloat2d, int m, int n) const;
    void AsNumpy2dFloat64(double* outdouble2d, int m, int n) const;
    void AsNumpy2dInt32(int* outint2d, int m, int n) const;
    void AsNumpy2dInt64(long long* outlong2d, int m, int n) const;

    bool operator==(MatrixBufferTemplate<T> const& other) const;

    void Print() const;

private:
    std::vector< T > mData;
    int mM;
    int mN;
};

template <class T>
MatrixBufferTemplate<T>::MatrixBufferTemplate()
: mData()
, mM(0)
, mN(0)
{
}

template <class T>
MatrixBufferTemplate<T>::MatrixBufferTemplate(int m, int n)
: mData( m*n )
, mM(m)
, mN(n)
{
}

template <class T>
MatrixBufferTemplate<T>::MatrixBufferTemplate(int m, int n, T value)
: mData( m*n, value )
, mM(m)
, mN(n)
{
}

template <class T>
MatrixBufferTemplate<T>::MatrixBufferTemplate(float* data, int m, int n)
: mData( m*n )
, mM(m)
, mN(n)
{
    for(int i=0; i<m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
MatrixBufferTemplate<T>::MatrixBufferTemplate(double* data, int m, int n)
: mData( m*n )
, mM(m)
, mN(n)
{
    for(int i=0; i<m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
MatrixBufferTemplate<T>::MatrixBufferTemplate(int* data, int m, int n)
: mData( m*n )
, mM(m)
, mN(n)
{
    for(int i=0; i<m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
MatrixBufferTemplate<T>::MatrixBufferTemplate(long long* data, int m, int n)
: mData( m*n )
, mM(m)
, mN(n)
{
    for(int i=0; i<m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
MatrixBufferTemplate<T>::~MatrixBufferTemplate()
{
}

template <class T>
void MatrixBufferTemplate<T>::Resize(int m, int n)
{
    Resize(m, n, T());
}

template <class T>
void MatrixBufferTemplate<T>::Resize(int m, int n, T value)
{
    if(static_cast<size_t>(m*n) > mData.size())
    {
        mData.resize(m*n, value);
    }
    mM = m;
    mN = n;
}

template <class T>
void MatrixBufferTemplate<T>::Zero()
{
    SetAll(static_cast<T>(0));
}

template <class T>
void MatrixBufferTemplate<T>::SetAll(const T value)
{
    std::fill(mData.begin(), mData.end(), value);
}

template <class T>
void MatrixBufferTemplate<T>::Set(int m, int n, T value)
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    mData[m*mN + n] = value;
}

template <class T>
T MatrixBufferTemplate<T>::Get(int m, int n) const
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return mData[m*mN + n];
}

template <class T>
void MatrixBufferTemplate<T>::SetUnsafe(int m, int n, T value)
{
    mData[m*mN + n] = value;
}

template <class T>
T MatrixBufferTemplate<T>::GetUnsafe(int m, int n) const
{
    return mData[ m*mN + n];
}

template <class T>
void MatrixBufferTemplate<T>::Incr(int m, int n, T value)
{
    mData[ m*mN + n] += value;
}

template <class T>
const T* MatrixBufferTemplate<T>::GetRowPtrUnsafe(int m) const
{
    return &mData[m*mN];
}

template <class T>
T MatrixBufferTemplate<T>::GetMax() const
{
    T max = std::numeric_limits<T>::min();
    for(int i=0; i<mM*mN; i++)
    {
        max = (max > mData[i]) ? max : mData[i];
    }
    return max;
}

template <class T>
T MatrixBufferTemplate<T>::GetMin() const
{
    T min = std::numeric_limits<T>::max();
    for(int i=0; i<mM*mN; i++)
    {
        min = (min < mData[i]) ? min : mData[i];
    }
    return min;
}

template <class T>
T MatrixBufferTemplate<T>::SumRow(int m) const
{
    T sum = Get(m,0);
    for(int c=1; c<mN; c++)
    {
        sum += Get(m,c);
    }
    return sum;
}

template <class T>
void MatrixBufferTemplate<T>::NormalizeRow(int m)
{
    T sum = SumRow(m);
    for(int c=0; c<mN; c++)
    {
        mData[ m*mN + c] /= sum;
    }
}

template <class T>
void MatrixBufferTemplate<T>::Append(const MatrixBufferTemplate<T>& buffer)
{
    ASSERT_ARG_DIM_1D(mN, buffer.GetN())
    const int oldM = mM;
    Resize(mM + buffer.GetM(), mN);
    for(int r=0; r<buffer.GetM(); r++)
    {
        for(int c=0; c<mN; c++)
        {
            Set(r+oldM, c, buffer.Get(r, c));
        }
    }
}

template <class T>
MatrixBufferTemplate<T> MatrixBufferTemplate<T>::Transpose() const
{
    MatrixBufferTemplate<T> transpose(mN, mM);
    for(int r=0; r<mM; r++)
    {
        for(int c=0; c<mN; c++)
        {
            transpose.Set(c, r, Get(r, c));
        }
    }
    return transpose;
}

template <class T>
MatrixBufferTemplate<T> MatrixBufferTemplate<T>::Slice(const VectorBufferTemplate<int>& indices) const
{
    MatrixBufferTemplate<T> sliced(indices.GetN(), mM);
    ASSERT_ARG_DIM_1D(indices.GetN(), 1)
    for(int i=0; i<indices.GetN(); i++)
    {
        int r = indices.Get(i);
        for(int c=0; c<mN; c++)
        {
            sliced.Set(i, c, Get(r, c));
        }
    }
    return sliced;
}

template <class T>
MatrixBufferTemplate<T> MatrixBufferTemplate<T>::SliceRow(const int row) const
{
    MatrixBufferTemplate<T> sliced(1, mN);
    for(int c=0; c<mN; c++)
    {
        sliced.Set(0, c, Get(row, c));
    }

    return sliced;
}

template <class T>
void MatrixBufferTemplate<T>::AsNumpy2dFloat32(float* outfloat2d, int m, int n) const
{
    ASSERT_ARG_DIM_2D(m, n, mM, mN)
    for(int i=0; i<m*n; i++)
    {
        outfloat2d[i] = static_cast<float>(mData[i]);
    }
}

template <class T>
void MatrixBufferTemplate<T>::AsNumpy2dFloat64(double* outdouble2d, int m, int n) const
{
    ASSERT_ARG_DIM_2D(m, n, mM, mN)
    for(int i=0; i<m*n; i++)
    {
        outdouble2d[i] = static_cast<double>(mData[i]);
    }
}


template <class T>
void MatrixBufferTemplate<T>::AsNumpy2dInt32(int* outint2d, int m, int n) const
{
    ASSERT_ARG_DIM_2D(m, n, mM, mN)
    for(int i=0; i<m*n; i++)
    {
        outint2d[i] = static_cast<int>(mData[i]);
    }
}

template <class T>
void MatrixBufferTemplate<T>::AsNumpy2dInt64(long long* outlong2d, int m, int n) const
{
    ASSERT_ARG_DIM_2D(m, n, mM, mN)
    for(int i=0; i<m*n; i++)
    {
        outlong2d[i] = static_cast<long long>(mData[i]);
    }
}

template<class T>
bool MatrixBufferTemplate<T>::operator==(MatrixBufferTemplate<T> const& other) const
{
    if (GetM() != other.GetM() || GetN() != other.GetN()) {
        return false;
    }

    return std::equal(mData.begin(), mData.end(), other.mData.begin());
}

template <class T>
void MatrixBufferTemplate<T>::Print() const
{
    std::cout << "[" << mM << " " << mN << "]" << std::endl;
    std::cout << "[" << std::endl;
    for(int m=0; m<mM; m++)
    {
        std::cout << "  [";
        for(int n=0; n<mN; n++)
        {
            std::cout << Get(m,n) << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

typedef MatrixBufferTemplate<float> Float32MatrixBuffer;
typedef MatrixBufferTemplate<double> Float64MatrixBuffer;
typedef MatrixBufferTemplate<int> Int32MatrixBuffer;
typedef MatrixBufferTemplate<long long> Int64MatrixBuffer;

Float32MatrixBuffer Float32Matrix2(float* float2d, int m, int n);
Float64MatrixBuffer Float64Matrix2(double* double2d, int m, int n);
Int32MatrixBuffer Int32Matrix2(int* int2d, int m, int n);
Int64MatrixBuffer Int64Matrix2(long long* long2d, int m, int n);

Float32MatrixBuffer Float32Matrix1(float* float1d, int n);
Float64MatrixBuffer Float64Matrix1(double* double1d, int n);
Int32MatrixBuffer Int32Matrix1(int* int1d, int n);
Int64MatrixBuffer Int64Matrix1(long long* long1d, int n);


