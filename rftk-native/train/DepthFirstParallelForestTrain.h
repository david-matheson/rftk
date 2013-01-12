#pragma once

#include "BufferCollection.h"
#include "FeatureExtractorI.h"
#include "BestSplitI.h"

class TrainingParams
{
public:
    TrainingParams();

    int mNumberOfJobs;

    // std::vector<FeatureExtractorI*> mFeatureExtractors;
    BufferCollection mData;
    FeatureExtractorI mFeatureExtractor;
    BestSplitI mBestSplit;

    int mNumberOfSamples;
    int mNumberOfSubsamples;
    bool mBootstrap;

};

class DepthFirstParallelForestTrain
{
public:

    DepthFirstParallelForestTrain( const TrainingParams& params ) {}
};