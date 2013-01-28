#pragma once

#include <tr1/memory>
#include <vector>

class Float32Tensor3Buffer {
public:
    Float32Tensor3Buffer() {}
    Float32Tensor3Buffer(int numberOfImgs, int m, int n);
    Float32Tensor3Buffer(float* data, int numberOfImgs, int m, int n);
    Float32Tensor3Buffer(double* data, int numberOfImgs, int m, int n);
    ~Float32Tensor3Buffer() {}

    int GetNumberOfImgs() const { return mNumberOfImgs; }
    int GetM() const { return mM; }
    int GetN() const { return mN; }

    void Set(int img, int m, int n, float value);
    float Get(int img, int m, int n) const;
    void SetUnsafe(int img, int m, int n, float value) { (*mData)[img*mM*mN + m*mN + n] = value; }
    float GetUnsafe(int img, int m, int n) const { return (*mData)[img*mM*mN + m*mN + n]; }
    // const float* GetDataRaw() const { return &mData[0]; }
    // const float* GetImgDataRaw(int img) const { return &mData[img*mM*mN]; }
    // void SetData(const float* data) { mData.assign(data, data + mNumberOfImgs*mM*mN); }

    const float* GetRowPtrUnsafe(int img, int m) const { return &(*mData)[img*mM*mN + m*mN]; }

    Float32Tensor3Buffer SharedMemoryCopy() { return *this; }
    void AsNumpy3dFloat32(float* outfloat3d, int l, int m, int n);

    void Print() const;

    std::tr1::shared_ptr< std::vector< float > > mData;
    int mNumberOfImgs;
    int mM;
    int mN;
};


class Int32Tensor3Buffer {
public:
    Int32Tensor3Buffer() {}
    Int32Tensor3Buffer(int numberOfImgs, int m, int n);
    Int32Tensor3Buffer(int* data, int numberOfImgs, int m, int n);
    Int32Tensor3Buffer(long long* data, int numberOfImgs, int m, int n);
    ~Int32Tensor3Buffer() {}

    int GetNumberOfImgs() const { return mNumberOfImgs; }
    int GetM() const { return mM; }
    int GetN() const { return mN; }

    void Set(int img, int m, int n, int value);
    int Get(int img, int m, int n) const;
    void SetUnsafe(int img, int m, int n, int value) { (*mData)[img*mM*mN + m*mN + n] = value; }
    int GetUnsafe(int img, int m, int n) const { return (*mData)[img*mM*mN + m*mN + n]; }
    // const int* GetDataRaw() const { return &mData[0]; }
    // const int* GetImgDataRaw(int img) const { return &mData[img*mM*mN]; }
    // void SetData(const int* data) { mData.assign(data, data + mNumberOfImgs*mM*mN); }

    const int* GetRowPtrUnsafe(int img, int m) const { return &(*mData)[img*mM*mN + m*mN]; }

    Int32Tensor3Buffer SharedMemoryCopy() { return *this; }
    void AsNumpy3dInt32(int* outint3d, int l, int m, int n);

    std::tr1::shared_ptr< std::vector< int > > mData;
    int mNumberOfImgs;
    int mM;
    int mN;
};


// Float32Tensor3Buffer Float32Tensor2(float* float2d, int m, int n);
Float32Tensor3Buffer Float32Tensor2(double* double2d, int m, int n);
// Int32Tensor3Buffer Int32Tensor2(int* int2d, int m, int n);
Int32Tensor3Buffer Int32Tensor2(long long* long2d, int m, int n);

// Float32Tensor3Buffer Float32Tensor3(float* float3d, int l, int m, int n);
Float32Tensor3Buffer Float32Tensor3(double* double3d, int l, int m, int n);
// Int32Tensor3Buffer Int32Tensor3(int* int3d, int l, int m, int n);
Int32Tensor3Buffer Int32Tensor3(long long* long3d, int l, int m, int n);