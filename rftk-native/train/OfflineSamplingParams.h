#pragma once

class OfflineSamplingParams
{
public:
    OfflineSamplingParams(int numberOfSamples, int numberOfSubsamples, bool bootstrap)
    : mNumberOfSamples(numberOfSamples)
    , mNumberOfSubsamples(numberOfSubsamples)
    , mBootstrap(bootstrap)
    {}

    int mNumberOfSamples;
    int mNumberOfSubsamples;
    bool mBootstrap;
};