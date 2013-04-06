#include "FeatureTypes.h"
#include "ImgFeatures.h"
#include <stdio.h>

int clampPixel(const int maxM, const int maxN, int* m, int *n)
{
    int clampedOffsets = 0;
    if(*m < 0)
    {
        clampedOffsets = 1;
        *m = 0;
    }
    if(*m >= maxM)
    {
        clampedOffsets = 1;
        *m = maxM-1;
    }
    if(*n < 0)
    {
        clampedOffsets = 1;
        *n = 0;
    }
    if(*n >= maxN)
    {
        clampedOffsets = 1;
        *n = maxN-1;
    }

    return clampedOffsets;
}


float pixelDepthDelta(const Float32Tensor3Buffer& depths, const int imgId, const int pixelM, const int pixelN,
                      const float ux, const float uy, const float vx, const float vy)
{
    const float scaleByDepth = 2.0f / depths.Get(imgId, pixelM, pixelN);

    int mU = pixelM + (int)(scaleByDepth * ux);
    int nU = pixelN + (int)(scaleByDepth * uy);
    int mV = pixelM + (int)(scaleByDepth * vx);
    int nV = pixelN + (int)(scaleByDepth * vy);

    clampPixel(depths.GetM(), depths.GetN(), &mU, &nU);
    clampPixel(depths.GetM(), depths.GetN(), &mV, &nV);

    float delta = depths.Get(imgId, mU, nU) - depths.Get(imgId, mV, nV);
    return delta;
}

bool pixelDepthDeltaTest(const Float32Tensor3Buffer& depths,
                          const Int32MatrixBuffer& pixelIndices,
                          const int index,
                          const Float32MatrixBuffer& offsets,
                          const int nodeIndex)
{
    const int imgId = pixelIndices.Get(index, 0);
    const int pixelM = pixelIndices.Get(index, 1);
    const int pixelN = pixelIndices.Get(index, 2);

    const float ux = offsets.Get(nodeIndex, 1);
    const float uy = offsets.Get(nodeIndex, 2);
    const float vx = offsets.Get(nodeIndex, 3);
    const float vy = offsets.Get(nodeIndex, 4);

    return (pixelDepthDelta(depths, imgId, pixelM, pixelN, ux, uy, vx, vy) > offsets.Get(nodeIndex, 0));
}



float pixelDepthEntangledYs( const float* depth, const int* perPixelNodeIds, const int maxM, const int maxN, const int pixelM, const int pixelN,
                                const float* nodeYs, const int yDim,
                                const float ux, const float uy,
                                const int componentId)
{
    const float scaleByDepth = 2.0f / depth[pixelM*maxN + pixelN];

    int mU = pixelM + (int)(scaleByDepth * ux);
    int nU = pixelN + (int)(scaleByDepth * uy);

    clampPixel(maxM, maxN, &mU, &nU);
    const int pixelNodeId = perPixelNodeIds[mU*maxN + nU];
    const float classProbability = nodeYs[pixelNodeId*yDim + componentId];
    return classProbability;
}

bool pixelDepthEntangledYsTest( const float* depth, const int* perPixelNodeIds, const int maxM, const int maxN, const int pixelM, const int pixelN,
                                const float* nodeYs, const int yDim,
                                const float* floatParams, const int* intParams)
{
    return (pixelDepthEntangledYs(depth, perPixelNodeIds, maxM, maxN, pixelM, pixelN, nodeYs, yDim, floatParams[1], floatParams[2], intParams[1]) > floatParams[0]);
}
