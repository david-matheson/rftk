#pragma once

#include <vector>
#include <map>
#include <set>
#include <utility>

#include "VectorBuffer.h"
#include "Forest.h"

///////////////////////////////////////////////////////////////////////////////
// This implements the priority queue for adding leafs to the fringe based on
// expected probability of error.  This is taken from
//
// Domingos, Pedro, and Geoff Hulten. "Mining high-speed data streams."
// Proceedings of the sixth ACM SIGKDD international conference on Knowledge
// discovery and data mining. ACM, 2000.
//
// http://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf
///////////////////////////////////////////////////////////////////////////////
class ProbabilityOfErrorFrontierQueue
{
public:
    ProbabilityOfErrorFrontierQueue(const int numberOfTrees);
    bool IsEmpty() const;
    void IncrDatapoints(const int treeIndex, long long count);
    std::pair<int, int> PopBest(const Forest& forest);
    void ProcessSplit(const Forest& forest, const int treeIndex,
                        const int nodeIndex, const int leftIndex, const int rightIndex);

private:
    std::vector<long long> mNumberDatapointsPerTree;
    std::map< std::pair<int, int>, long long> mNumberDatapointsOnNodeCreation;
    std::set< std::pair<int, int> > mQueuedFrontierLeaves;
};