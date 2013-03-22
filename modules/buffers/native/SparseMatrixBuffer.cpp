#include "SparseMatrixBuffer.h"

#define DEFINE_SWIG_INTERFACE_FUNCTION(TYPE_PREFIX, TYPE) \
TYPE_PREFIX ## SparseMatrixBuffer TYPE_PREFIX ## SparseMatrix(TYPE* values, int n_values, int* col, int n_col, int* rowPtr, int n_rowPtr, int n, int m) \
{ \
    return TYPE_PREFIX ## SparseMatrixBuffer(values, n_values, col, n_col, rowPtr, n_rowPtr, n, m); \
}

DEFINE_SWIG_INTERFACE_FUNCTION(Float32, float)
DEFINE_SWIG_INTERFACE_FUNCTION(Float64, double)
DEFINE_SWIG_INTERFACE_FUNCTION(Int32, int)
DEFINE_SWIG_INTERFACE_FUNCTION(Int64, long long)

