#pragma once

#include "buffers/VectorBuffer.h"
#include "buffers/MatrixBuffer.h"
#include "buffers/Tensor3Buffer.h"

#include "FeatureTypes.h"

int clampPixel(const int maxM, const int maxN, int* m, int *n);

float pixelDepthDelta(  const Float32Tensor3Buffer& depths, const int imgId, const int pixelM, const int pixelN,
                        const float ux, const float uy, const float vx, const float vy);

bool pixelDepthDeltaTest( const Float32Tensor3Buffer& depths,
                          const Int32MatrixBuffer& pixelIndices,
                          const int index,
                          const Float32MatrixBuffer& offsets,
                          const int nodeIndex);

float pixelDepthEntangledYs( const float* depth, const int* perPixelNodeIds, const int maxM, const int maxN, const int pixelM, const int pixelN,
                                const float* nodeYs, const int yDim,
                                const float ux, const float uy,
                                const int componentId);

bool pixelDepthEntangledYsTest( const float* depth, const int* perPixelNodeIds, const int maxM, const int maxN, const int pixelM, const int pixelN,
                                const float* nodeYs, const int yDim,
                                const float* floatParams, const int* intParams);
