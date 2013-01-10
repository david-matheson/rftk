#pragma once

#include "OnlineLeafI.h"

class AllThresholdsOnlineLeaf : public OnlineLeafI {
public:
  virtual void FitEpoch() {}
  virtual void GetYs(MatrixBufferFloat& result) {}
  virtual float GetBestImpurityValue() { return 0.0f; }
  virtual float GetSecondBestImpurityValue() { return 0.0f; }
};