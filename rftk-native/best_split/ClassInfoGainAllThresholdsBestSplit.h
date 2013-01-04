#pragma once

#include "MatrixBuffer.h"

#include "BestSplitI.h"


class ClassInfoGainAllThresholdsBestSplit : public BestSplitI {
public:
  ClassInfoGainAllThresholdsBestSplit(  const MatrixBufferInt& classlabels,
                                        const MatrixBufferFloat& sampleWeights,
                                        float ratioOfThresholdsToTest,
                                        int minNumberThresholdsToTest);

  ~ClassInfoGainAllThresholdsBestSplit();

  virtual void BestSplits(  const MatrixBufferInt& sampleIndices,
                            const MatrixBufferFloat& featureValues,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut);

private:
  MatrixBufferInt mClassLabels;
  MatrixBufferFloat mSampleWeights;
  float mRatioOfThresholdsToTest;
  int mMinNumberThresholdsToTest;
  int mMaxClass;
};