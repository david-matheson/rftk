#include <sstream>
#include <string>

#include "UniqueBufferId.h"

namespace UniqueBufferId
{

int globalBufferId = 0;

void Reset()
{
    globalBufferId = 0;
}

int GetId()
{
    // Note: this is not thread safe
    // GetId() must always be called from the main thread
    return ++globalBufferId;
}

std::string GetBufferString(const std::string& basename)
{
    std::ostringstream result;
    result << basename << GetId();
    return result.str();
}

} // namespace UniqueBufferId