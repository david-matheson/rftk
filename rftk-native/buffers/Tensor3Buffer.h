#pragma once

#include <vector>
#include <iostream>

#include "assert_util.h"

template <class T>
class Tensor3BufferTemplate {
public:
    Tensor3BufferTemplate() {}
    Tensor3BufferTemplate(int l, int m, int n);
    Tensor3BufferTemplate(float* data, int l, int m, int n);
    Tensor3BufferTemplate(double* data, int l, int m, int n);
    Tensor3BufferTemplate(int* data, int l, int m, int n);
    Tensor3BufferTemplate(long long* data, int l, int m, int n);
    ~Tensor3BufferTemplate();

    void Resize(int l, int m, int n);

    int GetL() const;
    int GetM() const;
    int GetN() const;

    void Set(int l, int m, int n, T value);
    T Get(int l, int m, int n) const;
    void SetUnsafe(int l, int m, int n, T value);
    T GetUnsafe(int l, int m, int n) const;

    const T* GetRowPtrUnsafe(int l, int m) const;
    void AppendSlice(const Tensor3BufferTemplate<T>& buffer);

    void AsNumpy3dFloat32(float* outfloat3d, int l, int m, int n) const;
    void AsNumpy3dInt32(int* outint3d, int l, int m, int n) const;

    void Print() const;

private:
    std::vector< T > mData;
    int mL;
    int mM;
    int mN;
};

template <class T>
Tensor3BufferTemplate<T>::Tensor3BufferTemplate(int l, int m, int n)
: mData(l*m*n)
, mL(l)
, mM(m)
, mN(n)
{
}

template <class T>
Tensor3BufferTemplate<T>::Tensor3BufferTemplate(float* data, int l, int m, int n)
: mData(l*m*n)
, mL(l)
, mM(m)
, mN(n)
{
    for(int i=0; i<l*m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
Tensor3BufferTemplate<T>::Tensor3BufferTemplate(double* data, int l, int m, int n)
: mData(l*m*n)
, mL(l)
, mM(m)
, mN(n)
{
    for(int i=0; i<l*m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
Tensor3BufferTemplate<T>::Tensor3BufferTemplate(int* data, int l, int m, int n)
: mData(l*m*n)
, mL(l)
, mM(m)
, mN(n)
{
    for(int i=0; i<l*m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
Tensor3BufferTemplate<T>::Tensor3BufferTemplate(long long* data, int l, int m, int n)
: mData(l*m*n)
, mL(l)
, mM(m)
, mN(n)
{
    for(int i=0; i<l*m*n; i++)
    {
        mData[i] = static_cast<T>(data[i]);
    }
}

template <class T>
Tensor3BufferTemplate<T>::~Tensor3BufferTemplate()
{
}

template <class T>
void Tensor3BufferTemplate<T>::Resize(int l, int m, int n)
{
    if(l*m*n > mData.size())
    {
        mData.resize(l*m*n);
    }
    mL = l;
    mM = m;
    mN = n;
}

template <class T>
int Tensor3BufferTemplate<T>::GetL() const 
{
    return mL; 
}

template <class T>
int Tensor3BufferTemplate<T>::GetM() const 
{
    return mM; 
}

template <class T>
int Tensor3BufferTemplate<T>::GetN() const 
{
    return mN; 
}

template <class T>
void Tensor3BufferTemplate<T>::Set(int l, int m, int n, T value)
{
    ASSERT_VALID_RANGE(l, 0, mL)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    mData[l*mM*mN + m*mN + n] = value;
}

template <class T>
T Tensor3BufferTemplate<T>::Get(int l, int m, int n) const
{
    ASSERT_VALID_RANGE(l, 0, mL)
    ASSERT_VALID_RANGE(m, 0, mM)
    ASSERT_VALID_RANGE(n, 0, mN)
    return mData[l*mM*mN + m*mN + n];
}

template <class T>
void Tensor3BufferTemplate<T>::SetUnsafe(int l, int m, int n, T value) 
{
    mData[l*mM*mN + m*mN + n] = value; 
}

template <class T>
T Tensor3BufferTemplate<T>::GetUnsafe(int l, int m, int n) const 
{
    return mData[l*mM*mN + m*mN + n]; 
}

template <class T>
const T* Tensor3BufferTemplate<T>::GetRowPtrUnsafe(int l, int m) const 
{
    ASSERT_VALID_RANGE(l, 0, mL)
    ASSERT_VALID_RANGE(m, 0, mM)
    return &mData[l*mM*mN + m*mN]; 
}

template <class T>
void Tensor3BufferTemplate<T>::AppendSlice(const Tensor3BufferTemplate<T>& buffer)
{
    ASSERT_ARG_DIM_2D(mM, mN, buffer.GetM(), buffer.GetN())
    const int oldL = mL;
    Resize(mL + buffer.GetL(), mM, mN);
    for(int l=0; l<buffer.GetL(); l++)
    {
        for(int m=0; m<buffer.GetM(); m++)
        {
            for(int n=0; n<mN; n++)
            {
                Set(l+oldL, m, n, buffer.Get(l, m, n));
            }
        }
    }
}

template <class T>
void Tensor3BufferTemplate<T>::AsNumpy3dFloat32(float* outfloat3d, int l, int m, int n) const
{
    ASSERT_ARG_DIM_3D(l, m, n, mL, mM, mN)
    for(int i=0; i<l*m*n; i++)
    {
        outfloat3d[i] = static_cast<float>(mData[i]);
    }
}

template <class T>
void Tensor3BufferTemplate<T>::AsNumpy3dInt32(int* outint3d, int l, int m, int n) const
{
    ASSERT_ARG_DIM_3D(l, m, n, mL, mM, mN)
    for(int i=0; i<l*m*n; i++)
    {
        outint3d[i] = static_cast<int>(mData[i]);
    }
}

template <class T>
void Tensor3BufferTemplate<T>::Print() const
{
    std::cout << "[" << mL << " " << mM << " " << mN << "]" << std::endl;
    std::cout << "[" << std::endl;
    for(int l=0; l<mL; l++)
    {
        std::cout << "  [" << std::endl;
        for(int m=0; m<mM; m++)
        {
            std::cout << "    [";
            for(int n=0; n<mN; n++)
            {
                std::cout << Get(l,m,n) << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}



typedef Tensor3BufferTemplate<float> Float32Tensor3Buffer;
typedef Tensor3BufferTemplate<int> Int32Tensor3Buffer;


// Float32Tensor3Buffer Float32Tensor2(float* float2d, int m, int n);
Float32Tensor3Buffer Float32Tensor2(double* double2d, int m, int n);
// Int32Tensor3Buffer Int32Tensor2(int* int2d, int m, int n);
Int32Tensor3Buffer Int32Tensor2(long long* long2d, int m, int n);

// Float32Tensor3Buffer Float32Tensor3(float* float3d, int l, int m, int n);
Float32Tensor3Buffer Float32Tensor3(double* double3d, int l, int m, int n);
// Int32Tensor3Buffer Int32Tensor3(int* int3d, int l, int m, int n);
Int32Tensor3Buffer Int32Tensor3(long long* long3d, int l, int m, int n);