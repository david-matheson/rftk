#pragma once

class OnlineSamplingParams
{
public:
    OnlineSamplingParams(const bool usePoisson, const float lambda)
    : mUsePoisson(usePoisson)
    , mLambda(lambda)
    {}

    bool mUsePoisson;
    float mLambda;
};