#pragma once

#include <vector>
#include <map>
#include <set>
#include <utility>
#include <algorithm>

#include <asserts.h>
#include <VectorBuffer.h>
#include <Forest.h>

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
template <class ProbabilityOfError>
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

template <class ProbabilityOfError>
ProbabilityOfErrorFrontierQueue<ProbabilityOfError>::ProbabilityOfErrorFrontierQueue(const int numberOfTrees)
: mNumberDatapointsPerTree(numberOfTrees)
, mNumberDatapointsOnNodeCreation()
, mQueuedFrontierLeaves()
{
    // Add the root node for all trees
    for(int treeIndex=0; treeIndex<numberOfTrees; treeIndex++)
    {
        mNumberDatapointsPerTree[treeIndex] = 0;
        std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, 0);
        mNumberDatapointsOnNodeCreation[treeNodeKey] = 0;
        mQueuedFrontierLeaves.insert(treeNodeKey);
    }
}

template <class ProbabilityOfError>
bool ProbabilityOfErrorFrontierQueue<ProbabilityOfError>::IsEmpty() const
{
    return (mQueuedFrontierLeaves.size() <= 0);
}

template <class ProbabilityOfError>
void ProbabilityOfErrorFrontierQueue<ProbabilityOfError>::IncrDatapoints(const int treeIndex, long long count)
{
    mNumberDatapointsPerTree[treeIndex] += count;
}

template <class ProbabilityOfError>
std::pair<int, int> ProbabilityOfErrorFrontierQueue<ProbabilityOfError>::PopBest(const Forest& forest)
{
    ASSERT(!IsEmpty())

    ProbabilityOfError probOfError;

    // Find node in queue with highest probability of improvement
    std::pair<int, int> maxProbabilityOfErrorTreeNodeKey = *mQueuedFrontierLeaves.begin();
    float maxProbabilityOfError = 0.0;
    typedef std::set< std::pair<int, int> >::iterator it_type;
    for(it_type iter = mQueuedFrontierLeaves.begin(); iter != mQueuedFrontierLeaves.end(); ++iter)
    {
        const int treeIndex = iter->first;
        const int nodeIndex = iter->second;
        const Tree& tree = forest.mTrees[treeIndex];
        const float pointsToReachNode = tree.GetCounts().Get(nodeIndex);
        const float pointsDuringNodeLifetime =
            static_cast<float>(mNumberDatapointsPerTree[treeIndex] - mNumberDatapointsOnNodeCreation[*iter]);
        const float probOfNode = pointsToReachNode / pointsDuringNodeLifetime;

        const float probOfErrorForNode = probOfError.ProbabilityOfError(tree, nodeIndex);

        const float probabilityOfError = probOfNode * probOfErrorForNode;
        if( probabilityOfError > maxProbabilityOfError )
        {
            maxProbabilityOfError = probabilityOfError;
            maxProbabilityOfErrorTreeNodeKey = *iter;
        }
    }

    mQueuedFrontierLeaves.erase(maxProbabilityOfErrorTreeNodeKey);
    return maxProbabilityOfErrorTreeNodeKey;
}

template <class ProbabilityOfError>
void ProbabilityOfErrorFrontierQueue<ProbabilityOfError>::ProcessSplit(const Forest& forest, const int treeIndex,
                    const int nodeIndex, const int leftIndex, const int rightIndex)
{
    // Initialize the mNumberDatapointsOnNodeCreation for the left and right child nodes
    // and add them to the queue.  To do this we need to calculate the appropriate
    // NumberDatapointsOnNodeCreation to ensure that probability of a datapoint reaching
    // the left or right leaf is correct.  The probability is factored in the following way
    //  P(left) = P(left|node)P(node)
    //  P(right) = P(right|node)P(node)
    //
    // where
    //  P(node) = #points_to_reach_node / (#points_since_node_creation)
    //  P(left|node) = #points_left / (#points_left + #points_right)
    //  P(right|node) = #points_right / (#points_left + #points_right)
    //
    // the total datapoints for the left and right splits required to maintain
    // the correct probability estimates are
    //  leftTotalDatapointsToMaintainProbEst = #points_left / P(left)
    //  rightTotalDatapointsToMaintainProbEst = #points_right / P(right)
    //
    // therefore the correct number of datapoints at creation is
    //  leftNumberDatapointsOnNodeCreation = currentNumberDatapoints - leftTotalDatapointsToMaintainProbEst
    //  rightNumberDatapointsOnNodeCreation = currentNumberDatapoints - rightTotalDatapointsToMaintainProbEst
    //
    // This may seem overly complex but it allows different candidate splits to start gathering
    // statistics at different times.  In otherwords, we allow for
    //  #points_since_node_creation != #points_left + #points_right
    //
    // Another approach would be for split mechanisms to maintain creation times but this only makes sense
    // in the online case.  Therefore, this complicated logic to estimate creation times seems like
    // the best approach.

    const Tree& tree = forest.mTrees[treeIndex];
    std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, nodeIndex);
    const float probabilityOfNode = tree.GetCounts().Get(nodeIndex) / static_cast<float>(mNumberDatapointsPerTree[treeIndex] - mNumberDatapointsOnNodeCreation[treeNodeKey]);

    const float probabilityOfLeftGivenNode = tree.GetCounts().Get(leftIndex) / (tree.GetCounts().Get(leftIndex) + tree.GetCounts().Get(rightIndex));
    const float probabilityOfLeft = probabilityOfNode * probabilityOfLeftGivenNode;
    const float leftTotalDatapointsToMaintainProbEst = tree.GetCounts().Get(leftIndex) / probabilityOfLeft;
    mNumberDatapointsOnNodeCreation[std::make_pair(treeIndex, leftIndex) ] = mNumberDatapointsPerTree[treeIndex] - leftTotalDatapointsToMaintainProbEst;
    mQueuedFrontierLeaves.insert( std::make_pair(treeIndex, leftIndex) );

    const float probabilityOfRightGivenNode = tree.GetCounts().Get(rightIndex) / (tree.GetCounts().Get(leftIndex) + tree.GetCounts().Get(rightIndex));
    const float probabilityOfRight = probabilityOfNode * probabilityOfRightGivenNode;
    const float rightTotalDatapointsToMaintainProbEst = tree.GetCounts().Get(rightIndex) / probabilityOfRight;
    mNumberDatapointsOnNodeCreation[std::make_pair(treeIndex, rightIndex) ] = mNumberDatapointsPerTree[treeIndex] - rightTotalDatapointsToMaintainProbEst;
    mQueuedFrontierLeaves.insert( std::make_pair(treeIndex, rightIndex) );
}