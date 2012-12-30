#include "../assert/assert.h"
#include "MatrixBuffer.h"

MatrixBufferFloat::MatrixBufferFloat(int m, int n)
: mData(m*n)
, mM(m)
, mN(n)
{
}

MatrixBufferFloat::MatrixBufferFloat(float* data, int m, int n)
: mData(data, data + m*n)
, mM(m)
, mN(n)
{
}

MatrixBufferFloat::MatrixBufferFloat(double* data, int m, int n)
: mData(m*n)
, mM(m)
, mN(n)
{
    for(int i=0; i<m*n; i++)
    {
        mData[i] = static_cast<float>(data[i]);
    }
}

void MatrixBufferFloat::Set(int m, int n, float value)
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    mData[m*mN + n] = value;
}

float MatrixBufferFloat::Get(int m, int n)
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return mData[m*mN + n];
}

void MatrixBufferFloat::AsNumpy(float* outfloat2d, int m, int n)
{
    ASSERT_ARG_DIM_2D(m, n, mM, mN)
    for(int i=0; i<m*n; i++)
    {
        outfloat2d[i] = mData[i];
    }    
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

void MatrixBufferInt::Set(int m, int n, int value)
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    mData[m*mN + n] = value;
}

int MatrixBufferInt::Get(int m, int n)
{
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return mData[m*mN + n];
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





