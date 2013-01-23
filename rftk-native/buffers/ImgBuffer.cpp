#include <cstdio>

#include "assert_util.h"
#include "ImgBuffer.h"


ImgBufferFloat::ImgBufferFloat(int numberOfImgs, int m, int n)
: mData( new std::vector< float >(numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

ImgBufferFloat::ImgBufferFloat(float* data, int numberOfImgs, int m, int n)
: mData( new std::vector< float >(data, data + numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

ImgBufferFloat::ImgBufferFloat(double* data, int numberOfImgs, int m, int n)
: mData( new std::vector< float >(numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
    std::tr1::shared_ptr<int> p(new int(42));
    for(int i=0; i<numberOfImgs*m*n; i++)
    {
        (*mData)[i] = static_cast<float>(data[i]);
    }
}

void ImgBufferFloat::Set(int img, int m, int n, float value)
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    (*mData)[img*mM*mN + m*mN + n] = value;
}

float ImgBufferFloat::Get(int img, int m, int n) const
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return (*mData)[img*mM*mN + m*mN + n];
}

void ImgBufferFloat::AsNumpy(float* outfloat3d, int l, int m, int n)
{
    ASSERT_ARG_DIM_3D(l, m, n, mNumberOfImgs, mM, mN)
    for(int i=0; i<l*m*n; i++)
    {
        outfloat3d[i] = (*mData)[i];
    }
}

void ImgBufferFloat::Print() const
{
    printf("[%d %d %d]\n", mNumberOfImgs, mM, mN);
    printf("[\n");
    for(int i=0; i<mNumberOfImgs; i++)
    {
        printf("  [\n");
        for(int m=0; m<mM; m++)
        {
            printf("    [");
            for(int n=0; n<mN; n++)
            {
                printf("%0.2f ", Get(i,m,n));
            }
            printf("]\n");
        }
        printf("  ]\n");
    }
    printf("]\n");
}


ImgBufferInt::ImgBufferInt(int numberOfImgs, int m, int n)
: mData( new std::vector< int >(numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

ImgBufferInt::ImgBufferInt(int* data, int numberOfImgs, int m, int n)
: mData( new std::vector< int >(data, data + numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

ImgBufferInt::ImgBufferInt(long long* data, int numberOfImgs, int m, int n)
: mData( new std::vector< int >(numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
    for(int i=0; i<numberOfImgs*m*n; i++)
    {
        (*mData)[i] = static_cast<int>(data[i]);
    }
}

void ImgBufferInt::Set(int img, int m, int n, int value)
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    (*mData)[img*mM*mN + m*mN + n] = value;
}

int ImgBufferInt::Get(int img, int m, int n) const
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return (*mData)[img*mM*mN + m*mN + n];
}

void ImgBufferInt::AsNumpy(int* outint3d, int l, int m, int n)
{
    ASSERT_ARG_DIM_3D(l, m, n, mNumberOfImgs, mM, mN)
    for(int i=0; i<l*m*n; i++)
    {
        outint3d[i] = (*mData)[i];
    }
}

// From 2d numpy arrays
ImgBufferFloat imgBufferFloat(float* float2d, int m, int n)
{
    return ImgBufferFloat(float2d, 1, m, n);
}

ImgBufferFloat imgBufferFloat64(double* double2d, int m, int n)
{
    return ImgBufferFloat(double2d, 1, m, n);
}

ImgBufferInt imgBufferInt(int* int2d, int m, int n)
{
    return ImgBufferInt(int2d, 1, m, n);
}

ImgBufferInt imgBufferInt64(long long* long2d, int m, int n)
{
    return ImgBufferInt(long2d, 1, m, n);
}

// From 3d numpy arrays
ImgBufferFloat imgsBufferFloat(float* float3d, int l, int m, int n)
{
    return ImgBufferFloat(float3d, l, m, n);
}

ImgBufferFloat imgsBufferFloat64(double* double3d, int l, int m, int n)
{
    return ImgBufferFloat(double3d, l, m, n);
}

ImgBufferInt imgsBufferInt(int* int3d, int l, int m, int n)
{
    return ImgBufferInt(int3d, l, m, n);
}

ImgBufferInt imgsBufferInt64(long long* long3d, int l, int m, int n)
{
    return ImgBufferInt(long3d, l, m, n);
}