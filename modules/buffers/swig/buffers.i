%module buffers
%{
    #define SWIG_FILE_WITH_INIT
    #include "VectorBuffer.h"
    #include "MatrixBuffer.h"
    #include "SparseMatrixBuffer.h"
    #include "Tensor3Buffer.h"
    #include "BufferCollection.h"
%}

%include <exception.i>
%include "numpy.i"
%import(module="rftk.asserts") "asserts.i"

%include "std_string.i"

%init %{
    import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {(float* float1d, int n)}
%apply (double* IN_ARRAY1, int DIM1) {(double* double1d, int n)}
%apply (int* IN_ARRAY1, int DIM1) {(int* int1d, int n)}
%apply (long long* IN_ARRAY1, int DIM1) {(long long* long1d, int n)}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* float2d, int m, int n)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* double2d, int m, int n)}
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int* int2d, int m, int n)}
%apply (long long* IN_ARRAY2, int DIM1, int DIM2) {(long long* long2d, int m, int n)}

%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* float3d, int l, int m, int n)}
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(double* double3d, int l, int m, int n)}
%apply (int* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(int* int3d, int l, int m, int n)}
%apply (long long* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(long long* long3d, int l, int m, int n)}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* outfloat1d, int n)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* outint1d, int n)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* outfloat2d, int m, int n)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* outint2d, int m, int n)}

%apply (float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* outfloat3d, int l, int m, int n)}
%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(int* outint3d, int l, int m, int n)}

%include "VectorBuffer.h"
%include "Tensor3Buffer.h"
%include "MatrixBuffer.h"

%template(Float32MatrixBuffer) MatrixBufferTemplate<float>;
%template(Float64MatrixBuffer) MatrixBufferTemplate<double>;
%template(Int32MatrixBuffer) MatrixBufferTemplate<int>;
%template(Int64MatrixBuffer) MatrixBufferTemplate<long long>;

%template(Float32Tensor3Buffer) Tensor3BufferTemplate<float>;
%template(Float64Tensor3Buffer) Tensor3BufferTemplate<double>;
%template(Int32Tensor3Buffer) Tensor3BufferTemplate<int>;
%template(Int64Tensor3Buffer) Tensor3BufferTemplate<long long>;

%template(Float32VectorBuffer) VectorBufferTemplate<float>;
%template(Float64VectorBuffer) VectorBufferTemplate<double>;
%template(Int32VectorBuffer) VectorBufferTemplate<int>;
%template(Int64VectorBuffer) VectorBufferTemplate<long long>;

/* Sparse matrix buffers */
%pythoncode %{
import scipy.sparse

def process_args_for_sparse_wrapper(*args):
    assert len(args) == 1
    S = args[0]
    if not scipy.sparse.issparse(S):
        raise TypeError("Must provide a sparse matrix.")

    if not scipy.sparse.isspmatrix_csr(S):
        S = S.tocsr()

    return (S.data, S.indices, S.indptr, S.shape[0], S.shape[1])
%}

%define DECLARE_WRAPPER_FOR_SPARSE_TYPE(ctype, function_name)
%apply (ctype* IN_ARRAY1, int DIM1) {(ctype* values, int n_values)}
%apply (int* IN_ARRAY1, int DIM1) {
    (int* col, int n_col),
    (int* rowPtr, int n_rowPtr)
    }
%pythonprepend function_name(ctype*, int, int*, int, int*, int, int, int) %{
    args = process_args_for_sparse_wrapper(*args)
%}
%enddef

DECLARE_WRAPPER_FOR_SPARSE_TYPE(float, Float32SparseMatrix)
DECLARE_WRAPPER_FOR_SPARSE_TYPE(double, Float64SparseMatrix)
DECLARE_WRAPPER_FOR_SPARSE_TYPE(int, Int32SparseMatrix)
DECLARE_WRAPPER_FOR_SPARSE_TYPE(long long, Int64SparseMatrix)

%include "SparseMatrixBuffer.h"

%template(Float32SparseMatrixBuffer) SparseMatrixBufferTemplate<float>;
%template(Float64SparseMatrixBuffer) SparseMatrixBufferTemplate<double>;
%template(Int32SparseMatrixBuffer) SparseMatrixBufferTemplate<int>;
%template(Int64SparseMatrixBuffer) SparseMatrixBufferTemplate<long long>;

%include "buffer_converters.i"

%include "BufferCollection.h"

%include "buffer_collection.i"

