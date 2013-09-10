#include <algorithm>

#include <asserts.h>
#include "Forest.h"
#include "ForestStats.h"

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

Forest::Forest(const Forest& rhs)
: mTrees(rhs.mTrees)
{
}

void Forest::AddForest(const Forest& forest)
{
    for(unsigned int i=0; i<forest.mTrees.size(); i++)
    {
        mTrees.push_back( forest.mTrees[i] );
    }
}

void Forest::AddTree(const Tree& tree)
{
    mTrees.push_back(tree);
}

void Forest::RemoveTree(const int index)
{
    mTrees.erase(mTrees.begin()+index);
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


ForestStats Forest::GetTreeStats(const int tree) const
{
    ForestStats stats;
    mTrees[tree].GatherStats(stats);
    return stats;
}