#include "VectorBuffer.h"

Float32VectorBuffer Float32Vector(double* double1d, int n)
{
   return Float32VectorBuffer(double1d, n); 
}

Float64VectorBuffer Float64Vector(double* double1d, int n)
{
    return Float64VectorBuffer(double1d, n);
}

Int32VectorBuffer Int32Vector(long long* long1d, int n)
{
    return Int32VectorBuffer(long1d, n); 
}

Int64VectorBuffer Int64Vector(long long* long1d, int n)
{
    return Int64VectorBuffer(long1d, n); 
}

