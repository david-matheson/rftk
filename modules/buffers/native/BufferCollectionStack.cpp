
#include "BufferCollectionStack.h"

BufferCollectionStack::BufferCollectionStack()
: mStack()
{
}

BufferCollectionStack::~BufferCollectionStack()
{
}

void BufferCollectionStack::Push(const BufferCollection* bufferCollection)
{
    mStack.push_front(bufferCollection);
}

void BufferCollectionStack::Pop()
{
    mStack.pop_front();
}