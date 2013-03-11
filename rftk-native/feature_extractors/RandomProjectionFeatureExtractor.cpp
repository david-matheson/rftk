#include "RandomProjectionFeatureExtractor.h"

#define DEFINE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(NAME) \
NAME ## RandomProjectionFeatureExtractorType NAME ## RandomProjectionFeatureExtractor \
    (int numberOfFeatures, \
     int numberOfComponents, \
     int numberOfComponentsInSubspace, \
     bool usePoisson) \
{ \
    return NAME ## RandomProjectionFeatureExtractorType \
        (numberOfFeatures, \
         numberOfComponents, \
         numberOfComponentsInSubspace, \
         usePoisson);                 \
}

DEFINE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float32);
DEFINE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float64);
DEFINE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float32Sparse);
DEFINE_RANDOM_PROJECTION_FEATURE_EXTRACTOR_FACTORY(Float64Sparse);
