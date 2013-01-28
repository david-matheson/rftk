#include "MatrixBuffer.h"



Float32MatrixBuffer Float32Matrix(float* float2d, int m, int n);
Float64MatrixBuffer Float64Matrix(double* double2d, int m, int n);
Int32MatrixBuffer Int32Matrix(int* int2d, int m, int n);
Int64MatrixBuffer Int64Matrix(long long* long2d, int m, int n);

Float32MatrixBuffer Float32Matrix(float* float2d, int m, int n)
{
    return Float32MatrixBuffer(float2d, m, n);
}

Float64MatrixBuffer Float64Matrix(double* double2d, int m, int n)
{
    return Float64MatrixBuffer(double2d, m, n);
}

Int32MatrixBuffer Int32Matrix(int* int2d, int m, int n)
{
    return Int32MatrixBuffer(int2d, m, n);
}

Int64MatrixBuffer Int64Matrix(long long* long2d, int m, int n)
{
    return Int64MatrixBuffer(long2d, m, n);
}



// Old helper

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





