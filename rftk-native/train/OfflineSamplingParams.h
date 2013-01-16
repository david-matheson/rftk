#pragma once

class OfflineSamplingParams
{
public:
    OfflineSamplingParams(int numberOfSamples, bool withReplacement)
    : mNumberOfSamples(numberOfSamples)
    , mWithReplacement(withReplacement)
    {}

    int mNumberOfSamples;
    bool mWithReplacement;
};