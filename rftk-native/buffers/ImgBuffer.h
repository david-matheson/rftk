#pragma once

#include <tr1/memory>
#include <vector>

class ImgBufferFloat {
public:
    ImgBufferFloat() {}
    ImgBufferFloat(int numberOfImgs, int m, int n);
    ImgBufferFloat(float* data, int numberOfImgs, int m, int n);
    ImgBufferFloat(double* data, int numberOfImgs, int m, int n);
    ~ImgBufferFloat() {}

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

    ImgBufferFloat SharedMemoryCopy() { return *this; }
    void AsNumpy(float* outfloat3d, int l, int m, int n);

    void Print() const;

    std::tr1::shared_ptr< std::vector< float > > mData;
    int mNumberOfImgs;
    int mM;
    int mN;
};


class ImgBufferInt {
public:
    ImgBufferInt() {}
    ImgBufferInt(int numberOfImgs, int m, int n);
    ImgBufferInt(int* data, int numberOfImgs, int m, int n);
    ImgBufferInt(long long* data, int numberOfImgs, int m, int n);
    ~ImgBufferInt() {}

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

    ImgBufferInt SharedMemoryCopy() { return *this; }
    void AsNumpy(int* outint3d, int l, int m, int n);

    std::tr1::shared_ptr< std::vector< int > > mData;
    int mNumberOfImgs;
    int mM;
    int mN;
};


ImgBufferFloat imgBufferFloat(float* float2d, int m, int n);
ImgBufferFloat imgBufferFloat64(double* double2d, int m, int n);
ImgBufferInt imgBufferInt(int* int2d, int m, int n);
ImgBufferInt imgBufferInt64(long long* long2d, int m, int n);

ImgBufferFloat imgsBufferFloat(float* float3d, int l, int m, int n);
ImgBufferFloat imgsBufferFloat64(double* double3d, int l, int m, int n);
ImgBufferInt imgsBufferInt(int* int3d, int l, int m, int n);
ImgBufferInt imgsBufferInt64(long long* long3d, int l, int m, int n);