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

#include <boost/lambda/lambda.hpp>

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
    SparseMatrixBufferTemplate(T const* values, int nV, size_t const* col, int nC, size_t const* rowPtr, int nRP, int m, int n);
    SparseMatrixBufferTemplate(std::vector<T> const& values, std::vector<size_t> const& col, std::vector<size_t> const& rowPtr, int m, int n);
    SparseMatrixBufferTemplate(T const* values, int m, int n); // construct fom dense data
    explicit SparseMatrixBufferTemplate(DenseType const& dense);
    

    ~SparseMatrixBufferTemplate() {}

    void Zero();

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

    void Append(const SparseMatrixBufferTemplate<T>& buffer);
    SparseMatrixBufferTemplate<T> Slice(const VectorBufferTemplate<int>& indices) const;
    SparseMatrixBufferTemplate<T> SliceRow(int row) const;

    bool operator==(SparseMatrixBufferTemplate<T> const& other);
    bool operator!=(SparseMatrixBufferTemplate<T> const& other)
    {
        return !(*this == other);
    }

    std::string ToString() const;
    void Print() const;

    void PrintInternalContents() const;

private:
    void priv_initFromDensePointer(T const* values, int m, int n);

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
SparseMatrixBufferTemplate<T>::SparseMatrixBufferTemplate(T const* values, int nV, size_t const* col, int nC, size_t const* rowPtr, int nRP, int m, int n)
    : mValues(values, values + nV)
    , mCol(col, col + nC)
    , mRowPtr(rowPtr, rowPtr + nRP)
    , mM(m)
    , mN(n)
{
}

template<typename T>
SparseMatrixBufferTemplate<T>::SparseMatrixBufferTemplate(std::vector<T> const& values, std::vector<size_t> const& col, std::vector<size_t> const& rowPtr, int m, int n)
    : mValues(values)
    , mCol(col)
    , mRowPtr(rowPtr)
    , mM(m)
    , mN(n)
{
}

template<typename T>
SparseMatrixBufferTemplate<T>::SparseMatrixBufferTemplate(T const* values, int m, int n)
{
    priv_initFromDensePointer(values, m, n);
}


template<typename T>
SparseMatrixBufferTemplate<T>::SparseMatrixBufferTemplate(DenseType const& dense)
{
    priv_initFromDensePointer(dense.GetRowPtrUnsafe(0), dense.GetM(), dense.GetN());
}

template<typename T>
void SparseMatrixBufferTemplate<T>::priv_initFromDensePointer(T const* values, int m, int n)
{
    mM = m;
    mN = n;

    // clear any current data
    mValues.clear();
    mCol.clear();
    mRowPtr.clear();

    T zero(0);
    size_t elementCounter = 0;
    for (int i=0; i<m; ++i) {
        mRowPtr.push_back(elementCounter);
        for (int j=0; j<n; ++j) {
            T const& value = values[i*n+j];
            if (value != zero) {
                mValues.push_back(value);
                mCol.push_back(j);
                elementCounter += 1;
            }
        }
    }
    mRowPtr.push_back(elementCounter);
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

    //return std::accumulate(mValues.begin() + valIndexBegin, mValues.begin() + valIndexEnd, T(0));
    return std::accumulate(&mValues[valIndexBegin], &mValues[valIndexEnd], T(0));
}


template<typename T>
void SparseMatrixBufferTemplate<T>::NormalizeRow(int m)
{
    ASSERT_VALID_RANGE(m, 0, mM);

    size_t valIndexBegin = mRowPtr[m];
    size_t valIndexEnd = mRowPtr[m+1];

    T Z = SumRow(m);

    // don't do anything if the row is all zeros
    if (Z != 0) {
        using namespace boost::lambda;
        std::transform(&mValues[valIndexBegin], &mValues[valIndexEnd],
                       &mValues[valIndexBegin], _1 / Z);
    }
}

template<typename T>
void SparseMatrixBufferTemplate<T>::Append(const SparseMatrixBufferTemplate<T>& other)
{
    ASSERT(mN == other.GetN() || mN == 0);

    if (mN == 0) {
        // if this matrix has no shape then appropriate the horizontal
        // shape from the thing that's being appended to us
        mN = other.mN;

        mValues.clear();
        mCol.clear();
        mRowPtr.clear();

        mRowPtr.push_back(0);
    }

    mM += other.mM;

    mValues.reserve(mValues.size() + other.mValues.size());
    mCol.reserve(mCol.size() + other.mCol.size());
    mRowPtr.reserve(mRowPtr.size() + other.mRowPtr.size());

    // copy the row indices, offsetting them to be after the rows of this matrix
    size_t base = mRowPtr.back();
    using namespace boost::lambda;
    // We already have the index where the next row will start in
    // mRowPtr so don't copy the first element of other.mRowPtr.
    std::transform(other.mRowPtr.begin() + 1, other.mRowPtr.end(),
                   std::back_inserter(mRowPtr), _1 + base);

    // column indices and values can just be copied directly
    std::copy(other.mCol.begin(), other.mCol.end(), std::back_inserter(mCol));
    std::copy(other.mValues.begin(), other.mValues.end(), std::back_inserter(mValues));
}

template<typename T>
SparseMatrixBufferTemplate<T> SparseMatrixBufferTemplate<T>::Slice(const VectorBufferTemplate<int>& indices) const
{
    // Don't use the constructor with a size since it sets up mRowPtr
    // for an empy matrix, we're going to build mRowPtr manually here.
    SparseMatrixBufferTemplate<T> sliced;
    sliced.mM = indices.GetN();
    sliced.mN = mN;

    size_t elementCount = 0;
    for (int i=0; i<indices.GetN(); ++i) {
        int index = indices.Get(i);
        ASSERT_VALID_RANGE(index, 0, mM);

        size_t rowBegin = mRowPtr[index];
        size_t rowEnd = mRowPtr[index+1];

        sliced.mRowPtr.push_back(elementCount);
        elementCount += rowEnd - rowBegin;

        std::copy(&mValues[rowBegin], &mValues[rowEnd], std::back_inserter(sliced.mValues));
        std::copy(&mCol[rowBegin], &mCol[rowEnd], std::back_inserter(sliced.mCol));
    }
    sliced.mRowPtr.push_back(elementCount);

    return sliced;

}

template<typename T>
SparseMatrixBufferTemplate<T> SparseMatrixBufferTemplate<T>::SliceRow(int m) const
{
    ASSERT_VALID_RANGE(m, 0, mM);

    size_t indexBegin = mRowPtr[m];
    size_t indexEnd = mRowPtr[m+1];

    size_t newRowPtr[] = {0, indexEnd - indexBegin};
    SparseMatrixBufferTemplate<T> sliced
        (&mValues[indexBegin], indexEnd - indexBegin,
         &mCol[indexBegin], indexEnd - indexBegin,
         &newRowPtr[0], 2,
         1, mN);

    return sliced;
}

template<typename T>
bool SparseMatrixBufferTemplate<T>::operator==(SparseMatrixBufferTemplate<T> const& other)
{
    bool same = true;
    same &= (mValues == other.mValues);
    same &= (mCol == other.mCol);
    same &= (mRowPtr == other.mRowPtr);
    same &= (mM == other.mM);
    same &= (mN == other.mN);
    return same;
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

template<typename T>
void SparseMatrixBufferTemplate<T>::PrintInternalContents() const
{
    std::cout << "mValues: ";
    std::copy(mValues.begin(), mValues.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\nmCol: ";
    std::copy(mCol.begin(), mCol.end(), std::ostream_iterator<size_t>(std::cout, " "));
    std::cout << "\nmRowPtr: ";
    std::copy(mRowPtr.begin(), mRowPtr.end(), std::ostream_iterator<size_t>(std::cout, " "));
    std::cout << std::endl;
}
