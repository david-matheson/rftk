#include <cstdio>

#include "assert_util.h"
#include "Tensor3Buffer.h"


Float32Tensor3Buffer::Float32Tensor3Buffer(int numberOfImgs, int m, int n)
: mData( new std::vector< float >(numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

Float32Tensor3Buffer::Float32Tensor3Buffer(float* data, int numberOfImgs, int m, int n)
: mData( new std::vector< float >(data, data + numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

Float32Tensor3Buffer::Float32Tensor3Buffer(double* data, int numberOfImgs, int m, int n)
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

void Float32Tensor3Buffer::Set(int img, int m, int n, float value)
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    (*mData)[img*mM*mN + m*mN + n] = value;
}

float Float32Tensor3Buffer::Get(int img, int m, int n) const
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return (*mData)[img*mM*mN + m*mN + n];
}

void Float32Tensor3Buffer::AsNumpy3dFloat32(float* outfloat3d, int l, int m, int n)
{
    ASSERT_ARG_DIM_3D(l, m, n, mNumberOfImgs, mM, mN)
    for(int i=0; i<l*m*n; i++)
    {
        outfloat3d[i] = (*mData)[i];
    }
}

void Float32Tensor3Buffer::Print() const
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


Int32Tensor3Buffer::Int32Tensor3Buffer(int numberOfImgs, int m, int n)
: mData( new std::vector< int >(numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

Int32Tensor3Buffer::Int32Tensor3Buffer(int* data, int numberOfImgs, int m, int n)
: mData( new std::vector< int >(data, data + numberOfImgs*m*n) )
, mNumberOfImgs(numberOfImgs)
, mM(m)
, mN(n)
{
}

Int32Tensor3Buffer::Int32Tensor3Buffer(long long* data, int numberOfImgs, int m, int n)
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

void Int32Tensor3Buffer::Set(int img, int m, int n, int value)
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    (*mData)[img*mM*mN + m*mN + n] = value;
}

int Int32Tensor3Buffer::Get(int img, int m, int n) const
{
    ASSERT_VALID_RANGE(img, 0, mNumberOfImgs)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return (*mData)[img*mM*mN + m*mN + n];
}

void Int32Tensor3Buffer::AsNumpy3dInt32(int* outint3d, int l, int m, int n)
{
    ASSERT_ARG_DIM_3D(l, m, n, mNumberOfImgs, mM, mN)
    for(int i=0; i<l*m*n; i++)
    {
        outint3d[i] = (*mData)[i];
    }
}

// From 2d numpy arrays
// Float32Tensor3Buffer Float32Tensor2(float* float2d, int m, int n)
// {
//     return Float32Tensor3Buffer(float2d, 1, m, n);
// }

Float32Tensor3Buffer Float32Tensor2(double* double2d, int m, int n)
{
    return Float32Tensor3Buffer(double2d, 1, m, n);
}

// Int32Tensor3Buffer Int32Tensor2(int* int2d, int m, int n)
// {
//     return Int32Tensor3Buffer(int2d, 1, m, n);
// }

Int32Tensor3Buffer Int32Tensor2(long long* long2d, int m, int n)
{
    return Int32Tensor3Buffer(long2d, 1, m, n);
}

// From 3d numpy arrays
// Float32Tensor3Buffer imgsBufferFloat(float* float3d, int l, int m, int n)
// {
//     return Float32Tensor3Buffer(float3d, l, m, n);
// }

Float32Tensor3Buffer Float32Tensor3(double* double3d, int l, int m, int n)
{
    return Float32Tensor3Buffer(double3d, l, m, n);
}

// Int32Tensor3Buffer imgsBufferInt(int* int3d, int l, int m, int n)
// {
//     return Int32Tensor3Buffer(int3d, l, m, n);
// }

Int32Tensor3Buffer Int32Tensor3(long long* long3d, int l, int m, int n)
{
    return Int32Tensor3Buffer(long3d, l, m, n);
}