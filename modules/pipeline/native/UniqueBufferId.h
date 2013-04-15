#pragma once

#include <string>

// ----------------------------------------------------------------------------
//
// Functions to create unique buffer names
//
// ----------------------------------------------------------------------------

namespace UniqueBufferId
{
    void Reset();
    int GetId();
    std::string GetBufferString(const std::string& basename);
}

