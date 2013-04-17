#pragma once

#include <list>
#include <asserts.h>

#include "BufferCollection.h"

// ----------------------------------------------------------------------------
//
// BufferCollectionStack is a stack of BufferCollections where Buffers from
// BufferCollections that are higher on the stack are returned first.  This is
// useful for scoping BufferCollections.  For example, first you want to
// find a buffer associated with a node.  If it doesn't exist you want to look
// down the stack at the BufferCollection associated with the tree and if it
// doesn't exist you look down the stack to buffer collection containing the
// static data.
//
// BufferCollectionStack does NOT own the memory of the BufferCollections that
// it wraps.
//
// ----------------------------------------------------------------------------

class BufferCollectionStack
{
public:
    BufferCollectionStack();
    ~BufferCollectionStack();

    void Push(const BufferCollection* bufferCollection);
    void Pop();

    template<typename BufferType>
    bool HasBuffer(BufferCollectionKey_t bufferKey) const;

    template<typename BufferType>
    BufferType const& GetBuffer(BufferCollectionKey_t bufferKey) const;

    template<typename BufferType>
    BufferType const* GetBufferPtr(BufferCollectionKey_t bufferKey) const;


private:
    std::list<const BufferCollection*> mStack;
};



template<typename BufferType>
bool BufferCollectionStack::HasBuffer(BufferCollectionKey_t bufferKey) const
{
    for (std::list<const BufferCollection*>::const_iterator it = mStack.begin() ; it != mStack.end(); ++it)
    {
        if((*it)->HasBuffer<BufferType>(bufferKey))
        {
            return true;
        }
    }
    return false;
}

template<typename BufferType>
BufferType const& BufferCollectionStack::GetBuffer(BufferCollectionKey_t bufferKey) const
{
    BufferType const* bufferPtr = GetBufferPtr<BufferType>(bufferKey);
    ASSERT(bufferPtr != NULL);
    return *bufferPtr;
}

template<typename BufferType>
BufferType const* BufferCollectionStack::GetBufferPtr(BufferCollectionKey_t bufferKey) const
{
    for (std::list<const BufferCollection*>::const_iterator it = mStack.begin(); it != mStack.end(); ++it)
    {
        if((*it)->HasBuffer<BufferType>(bufferKey))
        {
            return (*it)->GetBufferPtr<BufferType>(bufferKey);
        }
    }
    return NULL;
}