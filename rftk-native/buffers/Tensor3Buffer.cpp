#include "Tensor3Buffer.h"

#define DEFINE_SWIG_INTERFACE_FUNCTION_3D(TYPE_PREFIX, TYPE, TYPE_VAR) \
TYPE_PREFIX ## Tensor3Buffer TYPE_PREFIX ## Tensor3(TYPE* TYPE_VAR ## 3d, int l, int m, int n) \
{ \
    return TYPE_PREFIX ## Tensor3Buffer(TYPE_VAR ## 3d, l, m, n); \
}

DEFINE_SWIG_INTERFACE_FUNCTION_3D(Float32, float, float)
DEFINE_SWIG_INTERFACE_FUNCTION_3D(Float64, double, double)
DEFINE_SWIG_INTERFACE_FUNCTION_3D(Int32, int, int)
DEFINE_SWIG_INTERFACE_FUNCTION_3D(Int64, long long, long)

#define DEFINE_SWIG_INTERFACE_FUNCTION_2D(TYPE_PREFIX, TYPE, TYPE_VAR) \
TYPE_PREFIX ## Tensor3Buffer TYPE_PREFIX ## Tensor2(TYPE* TYPE_VAR ## 2d, int m, int n) \
{ \
    return TYPE_PREFIX ## Tensor3Buffer(TYPE_VAR ## 2d, 1, m, n); \
}

DEFINE_SWIG_INTERFACE_FUNCTION_2D(Float32, float, float)
DEFINE_SWIG_INTERFACE_FUNCTION_2D(Float64, double, double)
DEFINE_SWIG_INTERFACE_FUNCTION_2D(Int32, int, int)
DEFINE_SWIG_INTERFACE_FUNCTION_2D(Int64, long long, long)