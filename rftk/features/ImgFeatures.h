#pragma once

#include "FeatureTypes.h"

int clampPixel(const int maxM, const int maxN, int* m, int *n);

float pixelDepthDelta(  const float* depth, const int maxM, const int maxN, const int pixelM, const int pixelN,
                        const float ux, const float uy, const float vx, const float vy);

bool pixelDepthDeltaTest(  const float* depth, const int maxM, const int maxN, const int pixelM, const int pixelN,
                            const float* offsets);

float pixelDepthEntangledYs( const float* depth, const int* perPixelNodeIds, const int maxM, const int maxN, const int pixelM, const int pixelN,
                                const float* nodeYs, const int yDim,
                                const float ux, const float uy,
                                const int componentId);

bool pixelDepthEntangledYsTest( const float* depth, const int* perPixelNodeIds, const int maxM, const int maxN, const int pixelM, const int pixelN,
                                const float* nodeYs, const int yDim,
                                const float* floatParams, const int* intParams);
