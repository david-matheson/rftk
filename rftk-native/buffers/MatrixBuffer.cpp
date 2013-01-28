#include "MatrixBuffer.h"


Float32MatrixBuffer Float32Matrix(double* double2d, int m, int n)
{
    return Float32MatrixBuffer(double2d, m, n);
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

Float32MatrixBuffer vecBufferFloat(float* float1d, int m)
{
    return Float32MatrixBuffer(float1d, m, 1);
}

Float64MatrixBuffer vecBufferFloat64(double* double1d, int m)
{
    return Float64MatrixBuffer(double1d, m, 1);
}

Int32MatrixBuffer vecBufferInt(int* int1d, int m)
{
    return Int32MatrixBuffer(int1d, m, 1);
}

Int64MatrixBuffer vecBufferInt64(long long* long1d, int m)
{
    return Int64MatrixBuffer(long1d, m, 1);
}




