#pragma once

#include <vector>

#include "Tree.h"

class Forest
{
public:
    Forest();
    Forest( const std::vector<Tree>& trees );
    Forest( const int numberOfTrees );
    Forest( int numberOfTrees,
            int maxNumberNodes,
            int maxIntParamsDim,
            int maxFloatParamsDim,
            int maxYsDim );
    Forest(const Forest& rhs);

    int GetNumberOfTrees() const;
    Tree GetTree(const int index) const;
    ForestStats GetForestStats() const;
    ForestStats GetTreeStats(const int tree) const;

    std::vector<Tree> mTrees;
};