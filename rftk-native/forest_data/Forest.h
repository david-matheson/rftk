#pragma once

#include <vector>

#include "Tree.h"

class Forest
{
public:
    Forest() {}
    Forest( const std::vector<Tree>& trees );
    Forest( const int numberOfTrees );
    Forest( int numberOfTrees,
            int maxNumberNodes,
            int maxIntParamsDim,
            int maxFloatParamsDim,
            int maxYsDim );

    std::vector<Tree> mTrees;
};