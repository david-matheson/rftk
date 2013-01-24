#include <cfloat>
#include <climits>
#include <cstdio>


#include "assert_util.h"
#include "MatrixBuffer.h"

MatrixBufferFloat::MatrixBufferFloat()
: mData()
, mM(0)
, mN(0)
{
}

MatrixBufferFloat::MatrixBufferFloat(int m, int n)
: mData( m*n )
, mM(m)
, mN(n)
{
}

MatrixBufferFloat::MatrixBufferFloat(float* data, int m, int n)
: mData( data, data + m*n )
, mM(m)
, mN(n)
{
}

MatrixBufferFloat::MatrixBufferFloat(double* data, int m, int n)
: mData( m*n )
, mM(m)
, mN(n)
{
    for(int i=0; i<m*n; i++)
    {
        mData[i] = static_cast<float>(data[i]);
    }
}

void MatrixBufferFloat::Resize(int m, int n)
{
    if(m*n > mData.size())
    {
        mData.resize(m*n);
    }
    Zero();
    mM = m;
    mN = n;
}

void MatrixBufferFloat::AppendVertical(const MatrixBufferFloat& buffer)
{
    ASSERT_ARG_DIM_1D(mN, buffer.GetN())
    mData.resize((mM + buffer.GetM()) * mN);
    const int oldM = mM;
    mM += buffer.GetM();
    for(int r=0; r<buffer.GetM(); r++)
    {
        for(int c=0; c<mN; c++)
        {
            Set(r+oldM, c, buffer.Get(r, c));
        }
    }
}

MatrixBufferFloat MatrixBufferFloat::Transpose() const
{
    MatrixBufferFloat transpose(mN, mM);
    for(int r=0; r<mM; r++)
    {
        for(int c=0; c<mN; c++)
        {
            transpose.Set(c, r, Get(r, c));
        }
    }
    return transpose;
}

MatrixBufferFloat MatrixBufferFloat::Slice(const MatrixBufferInt& indices) const
{
    MatrixBufferFloat sliced(indices.GetM(), mN);
    ASSERT_ARG_DIM_1D(indices.GetN(), 1)
    for(int i=0; i<indices.GetM(); i++)
    {
        int r = indices.Get(i,0);
        for(int c=0; c<mN; c++)
        {
            sliced.Set(i, c, Get(r, c));
        }
    }
    return sliced;
}

void MatrixBufferFloat::Zero()
{
    SetAll(0.0);
}

void MatrixBufferFloat::SetAll(const float value)
{
    std::fill(mData.begin(), mData.end(), value);
}

void MatrixBufferFloat::Set(int m, int n, float value)
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    mData[m*mN + n] = value;
}

float MatrixBufferFloat::Get(int m, int n) const
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return mData[m*mN + n];
}

float MatrixBufferFloat::GetMax() const
{
    float max = FLT_MIN;
    for(int i=0; i<mM*mN; i++)
    {
        max = (max > mData[i]) ? max : mData[i];
    }
    return max;
}

float MatrixBufferFloat::GetMin() const
{
    float min = FLT_MAX;
    for(int i=0; i<mM*mN; i++)
    {
        min = (min < mData[i]) ? min : mData[i];
    }
    return min;
}

void MatrixBufferFloat::AsNumpy(float* outfloat2d, int m, int n)
{
    ASSERT_ARG_DIM_2D(m, n, mM, mN)
    for(int i=0; i<m*n; i++)
    {
        outfloat2d[i] = mData[i];
    }
}

void MatrixBufferFloat::Print() const
{
    printf("[%d %d]\n", mM, mN);
    printf("[\n");
    for(int m=0; m<mM; m++)
    {
        printf("  [");
        for(int n=0; n<mN; n++)
        {
            printf("%0.2f ", Get(m,n));
        }
        printf("]\n");
    }
    printf("]\n");
}

MatrixBufferInt::MatrixBufferInt()
: mData( )
, mM(0)
, mN(0)
{
}

MatrixBufferInt::MatrixBufferInt(int m, int n)
: mData(m*n)
, mM(m)
, mN(n)
{
}

MatrixBufferInt::MatrixBufferInt(int* data, int m, int n)
: mData(data, data + m*n)
, mM(m)
, mN(n)
{
}

MatrixBufferInt::MatrixBufferInt(long long* data, int m, int n)
: mData(m*n)
, mM(m)
, mN(n)
{
    for(int i=0; i<m*n; i++)
    {
        mData[i] = static_cast<int>(data[i]);
    }
}

void MatrixBufferInt::Resize(int m, int n)
{
    if(m*n > mData.size())
    {
        mData.resize(m*n);
    }
    Zero();
    mM = m;
    mN = n;
}

MatrixBufferInt::~MatrixBufferInt()
{
    // printf("MatrixBufferInt::~MatrixBufferInt [%d %d] %d %d", mM, mN, this, &mData);
}

void MatrixBufferInt::AppendVertical(const MatrixBufferInt& buffer)
{
    ASSERT_ARG_DIM_1D(mN, buffer.GetN())
    mData.resize((mM + buffer.GetM()) * mN);
    const int oldM = mM;
    mM += buffer.GetM();
    for(int r=0; r<buffer.GetM(); r++)
    {
        for(int c=0; c<mN; c++)
        {
            Set(r+oldM, c, buffer.Get(r, c));
        }
    }
}

MatrixBufferInt MatrixBufferInt::Transpose() const
{
    MatrixBufferInt transpose(mN, mM);
    for(int r=0; r<mM; r++)
    {
        for(int c=0; c<mN; c++)
        {
            transpose.Set(c, r, Get(r, c));
        }
    }
    return transpose;
}

MatrixBufferInt MatrixBufferInt::Slice(const MatrixBufferInt& indices) const
{
    MatrixBufferInt sliced(indices.GetM(), mN);
    ASSERT_ARG_DIM_1D(indices.GetN(), 1)
    for(int i=0; i<indices.GetM(); i++)
    {
        int r = indices.Get(i,0);
        for(int c=0; c<mN; c++)
        {
            sliced.Set(i, c, Get(r, c));
        }
    }
    return sliced;
}

void MatrixBufferInt::Zero()
{
    SetAll(0);
}

void MatrixBufferInt::SetAll(const int value)
{
    std::fill(mData.begin(), mData.end(), value);
}

void MatrixBufferInt::Set(int m, int n, int value)
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    mData[m*mN + n] = value;
}

int MatrixBufferInt::Get(int m, int n) const
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return mData[m*mN + n];
}

int MatrixBufferInt::GetMax() const
{
    int max = INT_MIN;
    for(int i=0; i<mM*mN; i++)
    {
        max = (max > mData[i]) ? max : mData[i];
    }
    return max;
}

int MatrixBufferInt::GetMin() const
{
    int min = INT_MAX;
    for(int i=0; i<mM*mN; i++)
    {
        min = (min < mData[i]) ? min : mData[i];
    }
    return min;
}

void MatrixBufferInt::AsNumpy(int* outint2d, int m, int n)
{
    ASSERT_ARG_DIM_2D(m, n, mM, mN)
    for(int i=0; i<m*n; i++)
    {
        outint2d[i] = mData[i];
    }
}

MatrixBufferFloat vecBufferFloat(float* float1d, int m)
{
    return MatrixBufferFloat(float1d, m, 1);
}

MatrixBufferFloat vecBufferFloat64(double* double1d, int m)
{
    return MatrixBufferFloat(double1d, m, 1);
}

MatrixBufferInt vecBufferInt(int* int1d, int m)
{
    return MatrixBufferInt(int1d, m, 1);
}

MatrixBufferInt vecBufferInt64(long long* long1d, int m)
{
    return MatrixBufferInt(long1d, m, 1);
}

MatrixBufferFloat matrixBufferFloat(float* float2d, int m, int n)
{
    return MatrixBufferFloat(float2d, m, n);
}

MatrixBufferFloat matrixBufferFloat64(double* double2d, int m, int n)
{
    return MatrixBufferFloat(double2d, m, n);
}

MatrixBufferInt matrixBufferInt(int* int2d, int m, int n)
{
    return MatrixBufferInt(int2d, m, n);
}

MatrixBufferInt matrixBufferInt64(long long* long2d, int m, int n)
{
    return MatrixBufferInt(long2d, m, n);
}





