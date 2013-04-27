#pragma once

#include <string>

// ----------------------------------------------------------------------------
//
// Functions to create unique buffer names
//
// ----------------------------------------------------------------------------

#define BufferId  std::string
#define NullKey "null"

void Reset();
int GetId();
BufferId GetBufferId(const BufferId& base);

