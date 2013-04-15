#include <sstream>
#include <string>

#include "UniqueBufferId.h"

namespace UniqueBufferId
{

int globalId = 0;

void Reset()
{
    globalId = 0;
}

int GetId()
{
    // Note: this is not thread safe
    // GetId() must always be called from the main thread
    return ++globalId;
}

BufferId GetBufferId(const BufferId& base)
{
    std::ostringstream result;
    result << base << GetId();
    return result.str();
}

} // namespace UniqueBufferId