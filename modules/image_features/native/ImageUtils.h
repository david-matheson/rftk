#pragma once

#include "Tensor3Buffer.h"

template <class BufferTypes>
void ClampPixel(const typename BufferTypes::Index maxM, 
                const typename BufferTypes::Index maxN, 
                typename BufferTypes::Index* m, 
                typename BufferTypes::Index* n)
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

template <class BufferTypes>
typename BufferTypes::FeatureValue
PixelDepthDelta(const Tensor3BufferTemplate<typename BufferTypes::SourceContinuous>& depths, 
                const typename BufferTypes::Index imgId, 
                const typename BufferTypes::Index pixelM, 
                const typename BufferTypes::Index pixelN,
                const typename BufferTypes::ParamsContinuous ux, 
                const typename BufferTypes::ParamsContinuous uy, 
                const typename BufferTypes::ParamsContinuous vx, 
                const typename BufferTypes::ParamsContinuous vy)
{
    const typename BufferTypes::ParamsContinuous scaleByDepth = typename BufferTypes::ParamsContinuous(2.0) / depths.Get(imgId, pixelM, pixelN);

    typename BufferTypes::Index  mU = pixelM + typename BufferTypes::Index(scaleByDepth * ux);
    typename BufferTypes::Index  nU = pixelN + typename BufferTypes::Index(scaleByDepth * uy);
    typename BufferTypes::Index  mV = pixelM + typename BufferTypes::Index(scaleByDepth * vx);
    typename BufferTypes::Index  nV = pixelN + typename BufferTypes::Index(scaleByDepth * vy);

    ClampPixel<BufferTypes>(depths.GetM(), depths.GetN(), &mU, &nU);
    ClampPixel<BufferTypes>(depths.GetM(), depths.GetN(), &mV, &nV);

    typename BufferTypes::FeatureValue delta = depths.Get(imgId, mU, nU) - depths.Get(imgId, mV, nV);
    return delta;
}