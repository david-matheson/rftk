#pragma once

#include <vector>

class MatrixBufferFloat {
public:
    MatrixBufferFloat() {}
    MatrixBufferFloat(int m, int n);
    MatrixBufferFloat(float* data, int m, int n);
    MatrixBufferFloat(double* data, int m, int n);
    ~MatrixBufferFloat() {}

    int GetM() const { return mM; }
    int GetN() const { return mN; }

    void Set(int m, int n, float value);
    float Get(int m, int n);
    void SetUnsafe(int m, int n, float value) { mData[m*mN + n] = value; }
    float GetUnsafe(int m, int n) { return mData[ m*mN + n]; }

    void AsNumpy(float* outfloat2d, int m, int n);

    std::vector< float > mData;
    int mM;
    int mN;
};


class MatrixBufferInt {
public:
    MatrixBufferInt() {}
    MatrixBufferInt(int m, int n);
    MatrixBufferInt(int* data, int m, int n);
    MatrixBufferInt(long long* data, int m, int n);
    ~MatrixBufferInt() {}

    int GetM() const { return mM; }
    int GetN() const { return mN; }

    void Set(int m, int n, int value);
    int Get(int m, int n);
    void SetUnsafe(int m, int n, int value) { mData[m*mN + n] = value; }
    int GetUnsafe(int m, int n) { return mData[ m*mN + n]; }

    void AsNumpy(int* outint2d, int m, int n);

    std::vector< int > mData;
    int mM;
    int mN;
};



MatrixBufferFloat vecBufferFloat(float* float1d, int m);
MatrixBufferFloat vecBufferFloat64(double* double1d, int m);
MatrixBufferInt vecBufferInt(int* int1d, int m);
MatrixBufferInt vecBufferInt64(long long* long1d, int m);

MatrixBufferFloat matrixBufferFloat(float* float2d, int m, int n);
MatrixBufferFloat matrixBufferFloat64(double* double2d, int m, int n);
MatrixBufferInt matrixBufferInt(int* int2d, int m, int n);
MatrixBufferInt matrixBufferInt64(long long* long2d, int m, int n);

