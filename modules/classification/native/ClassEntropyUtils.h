#pragma once

#include <cmath> 

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

template <class BufferTypes>
typename BufferTypes::SufficientStatsContinuous calcDiscreteEntropy(typename BufferTypes::SufficientStatsContinuous totalCounts, 
																	VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous> classHistogram, 
																	VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous> logClassHistogram)
{;
    const typename BufferTypes::SufficientStatsContinuous inverseTotalCounts = 1.0f / totalCounts;
    const typename BufferTypes::SufficientStatsContinuous logTotalCounts = log2(totalCounts);

    typename BufferTypes::SufficientStatsContinuous entropy = typename BufferTypes::SufficientStatsContinuous(0);
    for(int i=0; i<classHistogram.GetN() && totalCounts > 0.0f; i++)
    {
        const typename BufferTypes::SufficientStatsContinuous prob = inverseTotalCounts * classHistogram.Get(i);
        entropy -= prob * (logClassHistogram.Get(i) - logTotalCounts);
    }

    return entropy;
}