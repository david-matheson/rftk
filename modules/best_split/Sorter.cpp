#include <algorithm>

#include "Sorter.h"

Sorter::Sorter(const float* values, int numberOfValues)
: mValueIndices(numberOfValues)
{
    for(int i=0; i<numberOfValues; i++)
    {
        mValueIndices[i] = std::pair<float,int>(values[i], i);
    }
}

void Sorter::Sort()
{
    std::sort( mValueIndices.begin(), mValueIndices.end() );
}

int Sorter::GetUnSortedIndex(int sortedIndex)
{
    return mValueIndices[sortedIndex].second;
}

