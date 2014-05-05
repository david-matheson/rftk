#pragma once

#include "MatrixBuffer.h"
#include <iostream>

template <class SourceContinuousType,
        class SourceIntegerType, 
        class IndexType,
        class ParamsContinuousType,
        class ParamsIntegerType, 
        class FeatureValueType,
        class SufficientStatsContinuousType,
        class SufficientStatsIntegerType,
        class ImpurityValueType>
class BufferTypes {
public:
    BufferTypes() {}
    typedef SourceContinuousType SourceContinuous;
    typedef SourceIntegerType SourceInteger;
    typedef IndexType Index;
    typedef ParamsContinuousType ParamsContinuous;
    typedef ParamsIntegerType ParamsInteger;
    typedef FeatureValueType FeatureValue;
    typedef SufficientStatsContinuousType SufficientStatsContinuous;
    typedef SufficientStatsIntegerType SufficientStatsInteger;
    typedef ImpurityValueType ImpurityValue;
    typedef SourceContinuousType DatapointCounts;
    typedef SourceContinuousType TreeEstimator;
};

typedef BufferTypes<float, int, int, float, int, float, double, int, float> DefaultBufferTypes;
typedef BufferTypes<float, int, int, float, int, float, float, int, float> SinglePrecisionBufferTypes;