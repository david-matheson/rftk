#pragma once

#include <iostream>
#include <cstdio>

#include <string>
#include <sstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iterator>

#include "assert_util.h"
#include "MatrixBuffer.h"
#include "VectorBuffer.h"

template<typename T>
class SparseMatrixBufferTemplate {
public:
    typedef T ValueType;
    typedef MatrixBufferTemplate<T> DenseType;

public:
    SparseMatrixBufferTemplate();
    SparseMatrixBufferTemplate(int m, int n);
    SparseMatrixBufferTemplate(T* values, int nV, size_t* col, int nC, size_t* rowPtr, int nRP, int m, int n);

    ~SparseMatrixBufferTemplate() {}

    void Zero();
    void Resize(int m, int n);

    int GetM() const { return mM; }
    int GetN() const { return mN; }

    void Set(int m, int n, T value);
    T Get(int m, int n) const {
        T const* val = priv_valueAt(m, n);
        return val ? *val : T(0);
    }

    T GetMax() const;
    T GetMin() const;

    T SumRow(int m) const;
    void NormalizeRow(int m);

    // TODO: implement these
    //void AppendVertical(const SparseMatrixBufferTemplate<T>& buffer);
    //SparseMatrixBufferTemplate<T> Transpose() const;
    //SparseMatrixBufferTemplate<T> Slice(const VectorBufferTemplate<int>& indices) const;
    //SparseMatrixBufferTemplate<T> SliceRow(int row) const;

    std::string ToString() const;
    void Print() const;

private:
    T* priv_valueAt(int m, int n) {
        return const_cast<T*>(priv_valueAtConst(m, n));
    }

    T const* priv_valueAt(int m, int n) const {
        return priv_valueAtConst(m, n);
    }

    T const* priv_valueAtConst(int m, int n) const;

private:
    std::vector<T> mValues;
    std::vector<size_t> mCol;
    // mRowPtr has mM + 1 entries.  The last entry is the index at
    // which the next row (which doesn't exist) would begin.  Storing
    // this simplifies some of the calculations near the endge of the
    // array.
    std::vector<size_t> mRowPtr;
    int mM;
    int mN;
};

template<typename T>
SparseMatrixBufferTemplate<T>::SparseMatrixBufferTemplate()
    : mValues()
    , mCol()
    , mRowPtr()
    , mM(0)
    , mN(0)
{
}

template<typename T>
SparseMatrixBufferTemplate<T>::SparseMatrixBufferTemplate(int m, int n)
    : mValues()
    , mCol()
    , mRowPtr(m+1, T(0))
    , mM(m)
    , mN(n)
{
}

template<typename T>
SparseMatrixBufferTemplate<T>::SparseMatrixBufferTemplate(T* values, int nV, size_t* col, int nC, size_t* rowPtr, int nRP, int m, int n)
    : mValues(values, values + nV)
    , mCol(col, col + nC)
    , mRowPtr(rowPtr, rowPtr + nRP)
    , mM(m)
    , mN(n)
{
}

template<typename T>
void SparseMatrixBufferTemplate<T>::Resize(int m, int n)
{
    // FIXME: need to expand/shrink rowPtr
    mM = m;
    mN = n;
}

template<typename T>
void SparseMatrixBufferTemplate<T>::Zero()
{
    // clear and discard memory
    // http://www.cplusplus.com/reference/vector/vector/clear/
    std::vector<T>().swap(mValues);
    std::vector<size_t>().swap(mCol);
    std::vector<size_t>().swap(mRowPtr);
}


template<typename T>
T const* SparseMatrixBufferTemplate<T>::priv_valueAtConst(int m, int n) const
{
    ASSERT_VALID_RANGE(m, 0, mM);
    ASSERT_VALID_RANGE(n, 0, mN);

    size_t colIndexBegin = mRowPtr[m];
    size_t colIndexEnd = mRowPtr[m+1];

    // (m, :) is zero
    if (colIndexBegin == colIndexEnd) {
        return static_cast<T const*>(0);
    }

    std::vector<size_t>::const_iterator colIter = std::lower_bound(mCol.begin() + colIndexBegin, mCol.begin() + colIndexEnd, n);
    size_t valIndex = colIter - mCol.begin();

    // (m, n) is zero
    if (mCol[valIndex] != n) {
        return static_cast<T const*>(0);
    }

    return &mValues[valIndex];
}


template<typename T>
T SparseMatrixBufferTemplate<T>::GetMax() const
{
    T maxVal = *std::max_element(mValues.begin(), mValues.end());

    if (maxVal < T(0) && mValues.size() < mM*mN) {
        return T(0);
    }
    else {
        return maxVal;
    }
}

template<typename T>
T SparseMatrixBufferTemplate<T>::GetMin() const
{
    T minVal = *std::min_element(mValues.begin(), mValues.end());
    
    if (minVal > T(0) && mValues.size() < mM*mN) {
        return T(0);
    }
    else {
        return minVal;
    }
}

template<typename T>
T SparseMatrixBufferTemplate<T>::SumRow(int m) const
{
    ASSERT_VALID_RANGE(m, 0, mM);
    
    size_t valIndexBegin = mRowPtr[m];
    size_t valIndexEnd = mRowPtr[m+1];

    return std::accumulate(mValues.begin() + valIndexBegin, mValues.begin() + valIndexEnd, T(0));
}


template<typename T>
struct op_normalize {
    T mZ;
    op_normalize(T Z): mZ(Z) {}
    T operator()(T a) { return a/mZ; }
};
template<typename T>
void SparseMatrixBufferTemplate<T>::NormalizeRow(int m)
{
    ASSERT_VALID_RANGE(m, 0, mM);

    size_t valIndexBegin = mRowPtr[m];
    size_t valIndexEnd = mRowPtr[m+1];

    std::transform(mValues.begin() + valIndexBegin, mValues.begin() + valIndexEnd,
                   mValues.begin() + valIndexBegin, op_normalize<T>(SumRow(m)));
}

template<typename T>
std::string SparseMatrixBufferTemplate<T>::ToString() const 
{
    std::stringstream ss;

    for (int i=0; i<mM; ++i) {
        for (int j=0; j<mN; ++j) {
            T val = Get(i,j);
            if (val == 0) {
                ss << ".. ";
            }
            else {
                ss << val << " ";
            }
        }
        ss << "\n";
    }
    return ss.str();
}

template<typename T>
void SparseMatrixBufferTemplate<T>::Print() const
{
    std::cout << ToString();
}
