#pragma once

#include <vector>
#include <utility>

class Sorter
{
public:
    Sorter(const float* values, int numberOfValues);
    void Sort();
    int GetUnSortedIndex(int sortedIndex);

private:
    std::vector< std::pair<float, int> > mValueIndices;
};