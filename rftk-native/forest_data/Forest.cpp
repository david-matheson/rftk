#include "assert_util.h"
#include "Forest.h"

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
{
    for(int i=0; i<numberOfTrees; i++)
    {
        mTrees.push_back(Tree(maxNumberNodes, maxIntParamsDim, maxFloatParamsDim, maxYsDim));
    }
}