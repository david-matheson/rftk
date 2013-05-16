#include <stdio.h>

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

void BufferCollectionStack::Print() const
{
    int level = 0;
    for (std::list<const BufferCollection*>::const_iterator it = mStack.begin(); it != mStack.end(); ++it, ++level)
    {
        printf("-- Stack level %d --\n", level);
        (*it)->Print();
    }
    printf("--------------------\n");
}