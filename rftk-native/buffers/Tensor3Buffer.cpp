
#include "Tensor3Buffer.h"

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