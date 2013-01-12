#pragma once

#include <vector>

#include "Tree.h"

class Forest
{
public:
    Forest() {}
    Forest( const std::vector<Tree>& trees );

    std::vector<Tree> mTrees;
};