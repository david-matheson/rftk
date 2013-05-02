#pragma once

#include "Tensor3Buffer.h"

template <class IntType>
void ClampPixel(const IntType maxM, const IntType maxN, IntType* m, IntType *n)
{
    if(*m < 0)
    {
        *m = 0;
    }
    if(*m >= maxM)
    {
        *m = maxM-1;
    }
    if(*n < 0)
    {
        *n = 0;
    }
    if(*n >= maxN)
    {
        *n = maxN-1;
    }
}

template <class FloatType, class IntType>
FloatType PixelDepthDelta(const Tensor3BufferTemplate<FloatType>& depths, const IntType imgId, const IntType pixelM, const IntType pixelN,
                          const FloatType ux, const FloatType uy, const FloatType vx, const FloatType vy)
{
    const FloatType scaleByDepth = FloatType(2.0) / depths.Get(imgId, pixelM, pixelN);

    IntType mU = pixelM + IntType(scaleByDepth * ux);
    IntType nU = pixelN + IntType(scaleByDepth * uy);
    IntType mV = pixelM + IntType(scaleByDepth * vx);
    IntType nV = pixelN + IntType(scaleByDepth * vy);

    ClampPixel<IntType>(depths.GetM(), depths.GetN(), &mU, &nU);
    ClampPixel<IntType>(depths.GetM(), depths.GetN(), &mV, &nV);

    FloatType delta = depths.Get(imgId, mU, nU) - depths.Get(imgId, mV, nV);
    return delta;
}