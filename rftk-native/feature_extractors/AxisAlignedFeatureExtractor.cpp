#include "AxisAlignedFeatureExtractor.h"

#define DEFINE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(NAME) \
NAME ## AxisAlignedFeatureExtractorType NAME ## AxisAlignedFeatureExtractor \
    (int numberOfFeatures, \
     int numberOfComponents, \
     bool usePoisson) \
{ \
    return NAME ## AxisAlignedFeatureExtractorType( \
        numberOfFeatures, \
        numberOfComponents, \
        usePoisson); \
}

DEFINE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float32);
DEFINE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float64);
DEFINE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float32Sparse);
DEFINE_AXIS_ALIGNED_FEATURE_EXTRACTOR_FACTORY(Float64Sparse);
