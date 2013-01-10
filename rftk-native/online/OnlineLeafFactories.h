#pragma once

#include "OnlineLeafFactoryI.h"
// class MatrixBufferInt;
// class MatrixBufferFloat;

class AllThresholdsLeafFactory : OnlineLeafFactoryI {
public:
  virtual OnlineLeafI* Construct() { return NULL; }
};
