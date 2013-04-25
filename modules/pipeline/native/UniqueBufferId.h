#pragma once

#include <string>

// ----------------------------------------------------------------------------
//
// Functions to create unique buffer names
//
// ----------------------------------------------------------------------------

namespace UniqueBufferId
{
    typedef std::string BufferId;

    const BufferId NullKey("null");

    void Reset();
    int GetId();
    BufferId GetBufferId(const BufferId& base);
}

