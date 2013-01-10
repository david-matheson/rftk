#pragma once

#include "MatrixBuffer.h"


class OnlineLeafSet {
public:
  OnlineLeafSet() {}
  ~OnlineLeafSet() {}

  void FitEpoch(const MatrixBufferFloat& x, const MatrixBufferFloat& weigths, const MatrixBufferInt& ys);
  void ReadyToSplit(const MatrixBufferInt& leafIds) {}

  void AddLeaf() {}
  void RemoveLeaf(int leafId) {}

  // std::vector<> mActiveLeafs;

};