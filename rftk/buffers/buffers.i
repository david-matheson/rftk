%module "buffers"
%{
    #define SWIG_FILE_WITH_INIT
    #include "ImgBuffer.h"
    #include "MatrixBuffer.h"
%}

%include "numpy.i"
%include "../assert/assert.i"
%include "std_vector.i"

%include exception.i

%init %{
    import_array();
%}


%apply (float* IN_ARRAY1, int DIM1) {(float* float1d, int m)}
%apply (double* IN_ARRAY1, int DIM1) {(double* double1d, int m)}
%apply (int* IN_ARRAY1, int DIM1) {(int* int1d, int m)}
%apply (long long* IN_ARRAY1, int DIM1) {(long long* long1d, int m)}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* float2d, int m, int n)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* double2d, int m, int n)}
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int* int2d, int m, int n)}
%apply (long long* IN_ARRAY2, int DIM1, int DIM2) {(long long* long2d, int m, int n)}

%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* float3d, int l, int m, int n)}
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(double* double3d, int l, int m, int n)}
%apply (int* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(int* int3d, int l, int m, int n)}
%apply (long long* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(long long* long3d, int l, int m, int n)}

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* outfloat2d, int m, int n)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* outint2d, int m, int n)}

%apply (float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* outfloat3d, int l, int m, int n)}
%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(int* outint3d, int l, int m, int n)}


%include "ImgBuffer.h"
%include "MatrixBuffer.h"
