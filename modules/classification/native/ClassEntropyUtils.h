#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

template <class FloatType>
float calcDiscreteEntropy(FloatType totalCounts, VectorBufferTemplate<FloatType> classHistogram, VectorBufferTemplate<FloatType> logClassHistogram)
{;
    const FloatType inverseTotalCounts = 1.0f / totalCounts;
    const float logTotalCounts = log2(totalCounts);

    FloatType entropy = FloatType(0);
    for(int i=0; i<classHistogram.GetN() && totalCounts > 0.0f; i++)
    {
        const FloatType prob = inverseTotalCounts * classHistogram.Get(i);
        entropy -= prob * (logClassHistogram.Get(i) - logTotalCounts);
    }

    return entropy;
}