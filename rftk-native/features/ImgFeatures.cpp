#include "PixelFeatures.h"
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


float pixelDepthDelta(const float* depth, const int maxM, const int maxN, const int pixelM, const int pixelN,
                        const float ux, const float uy, const float vx, const float vy)
{

    const float scaleByDepth = 2.0f / depth[pixelM*maxN + pixelN];

    int mU = pixelM + (int)(scaleByDepth * ux);
    int nU = pixelN + (int)(scaleByDepth * uy);
    int mV = pixelM + (int)(scaleByDepth * vx);
    int nV = pixelN + (int)(scaleByDepth * vy);

    const int clampedOffsetsU = clampPixel(maxM, maxN, &mU, &nU);
    const int clampedOffsetsV = clampPixel(maxM, maxN, &mV, &nV);

    float delta = depth[mU*maxN + nU] - depth[mV*maxN + nV];
    return delta;
}

bool pixelDepthDeltaTest(const float* depth, const int maxM, const int maxN, const int pixelM, const int pixelN,
                        const float* offsets)
{
    return (pixelDepthDelta(depth, maxM, maxN, pixelM, pixelN, offsets[1], offsets[2], offsets[3], offsets[4]) > offsets[0]);
}



float pixelDepthEntangledYs( const float* depth, const int* perPixelNodeIds, const int maxM, const int maxN, const int pixelM, const int pixelN,
                                const float* nodeYs, const int yDim,
                                const float ux, const float uy,
                                const int componentId)
{
    const float scaleByDepth = 2.0f / depth[pixelM*maxN + pixelN];

    int mU = pixelM + (int)(scaleByDepth * ux);
    int nU = pixelN + (int)(scaleByDepth * uy);

    const int clampedOffsetsU = clampPixel(maxM, maxN, &mU, &nU);
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
