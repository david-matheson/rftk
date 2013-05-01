#pragma once

#include <string>

// ----------------------------------------------------------------------------
//
// Functions to create unique buffer names
//
// ----------------------------------------------------------------------------

typedef std::string BufferId;
#define NullKey "null"

void Reset();
int GetId();
BufferId GetBufferId(const BufferId& base);

