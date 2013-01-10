#pragma once

#include "OnlineLeafI.h"
// class MatrixBufferInt;
// class MatrixBufferFloat;

class OnlineLeafFactoryI {
public:
  virtual OnlineLeafI* Construct() { return NULL; }
};
