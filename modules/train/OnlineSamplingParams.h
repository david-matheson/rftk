#pragma once

class OnlineSamplingParams
{
public:
    OnlineSamplingParams(const bool usePoisson, const float lambda, const int evalSplitPeriod=1)
    : mUsePoisson(usePoisson)
    , mLambda(lambda)
    , mEvalSplitPeriod(evalSplitPeriod)
    {}

    const bool mUsePoisson;
    const float mLambda;
    const int mEvalSplitPeriod;
};