#include "VectorBuffer.h"

#define DEFINE_SWIG_INTERFACE_FUNCTION_1D(TYPE_PREFIX, TYPE, TYPE_VAR) \
TYPE_PREFIX ## VectorBuffer TYPE_PREFIX ## Vector(TYPE* TYPE_VAR ## 1d, int n) \
{ \
    return TYPE_PREFIX ## VectorBuffer(TYPE_VAR ## 1d, n); \
}

DEFINE_SWIG_INTERFACE_FUNCTION_1D(Float32, float, float)
DEFINE_SWIG_INTERFACE_FUNCTION_1D(Float64, double, double)
DEFINE_SWIG_INTERFACE_FUNCTION_1D(Int32, int, int)
DEFINE_SWIG_INTERFACE_FUNCTION_1D(Int64, long long, long)