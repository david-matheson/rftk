#include <algorithm>

#include <asserts.h>
#include "Forest.h"

Forest::Forest()
: mTrees(0)
{
}

Forest::Forest( const std::vector<Tree>& trees )
: mTrees(trees)
{
}

Forest::Forest( const int numberOfTrees )
: mTrees( numberOfTrees )
{
}

Forest::Forest( int numberOfTrees,
                int maxNumberNodes,
                int maxIntParamsDim,
                int maxFloatParamsDim,
                int maxYsDim )
: mTrees(numberOfTrees)
{
    for(int i=0; i<numberOfTrees; i++)
    {
        mTrees[i] = Tree(maxNumberNodes, maxIntParamsDim, maxFloatParamsDim, maxYsDim);
    }
}

int Forest::GetNumberOfTrees() const
{
    return mTrees.size();
}


Tree Forest::GetTree(const int index) const
{
    return mTrees[index];
}

ForestStats Forest::GetForestStats() const
{
    ForestStats stats;
    for(unsigned int i=0; i<mTrees.size(); i++)
    {
        mTrees[i].GatherStats(stats);
    }
    return stats;
}


