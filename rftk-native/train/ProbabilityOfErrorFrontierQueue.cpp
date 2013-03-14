#include <algorithm>

#include "assert_util.h"
#include "ProbabilityOfErrorFrontierQueue.h"

ProbabilityOfErrorFrontierQueue::ProbabilityOfErrorFrontierQueue(const int numberOfTrees)
: mNumberDatapointsPerTree(numberOfTrees)
, mNumberDatapointsOnNodeCreation()
, mQueuedFrontierLeaves()
{
    // Add the root node for all trees
    for(int treeIndex=0; treeIndex<numberOfTrees; treeIndex++)
    {
        std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, 0);
        mNumberDatapointsOnNodeCreation[treeNodeKey] = 0;
        mQueuedFrontierLeaves.insert(treeNodeKey);
    }
}

bool ProbabilityOfErrorFrontierQueue::IsEmpty() const
{
    return (mQueuedFrontierLeaves.size() > 0);
}

void ProbabilityOfErrorFrontierQueue::IncrDatapoints(const int treeIndex, long long count)
{
    mNumberDatapointsPerTree[treeIndex] += count;
}

std::pair<int, int> ProbabilityOfErrorFrontierQueue::PopBest(const Forest& forest)
{
    ASSERT(!IsEmpty())
    typedef std::set< std::pair<int, int> >::iterator it_type;

    // Find node in queue with highest probability of improvement
    std::pair<int, int> maxProbabilityOfErrorTreeNodeKey = *mQueuedFrontierLeaves.begin();
    float maxProbabilityOfError = 0.0;
    for(it_type iter = mQueuedFrontierLeaves.begin(); iter != mQueuedFrontierLeaves.end(); ++iter)
    {
        const Tree& tree = forest.mTrees[iter->first];
        const float pointsToReachNode = tree.mCounts.Get(iter->second);
        const float pointsDuringNodeLifetime =
            static_cast<float>(mNumberDatapointsPerTree[iter->first] - mNumberDatapointsOnNodeCreation[*iter]);
        const float probOfNode = pointsToReachNode / pointsDuringNodeLifetime;

        const float* ys = tree.mYs.GetRowPtrUnsafe(iter->second);
        const float probOfErrorForNode = 1.0f - *std::max_element(ys, ys + tree.mYs.GetN());

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

void ProbabilityOfErrorFrontierQueue::ProcessSplit(const Forest& forest, const int treeIndex,
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
    const float probabilityOfNode = tree.mCounts.Get(nodeIndex) / static_cast<float>(mNumberDatapointsPerTree[treeIndex] - mNumberDatapointsOnNodeCreation[treeNodeKey]);

    const float probabilityOfLeftGivenNode = tree.mCounts.Get(leftIndex) / (tree.mCounts.Get(leftIndex) + tree.mCounts.Get(rightIndex));
    const float probabilityOfLeft = probabilityOfNode * probabilityOfLeftGivenNode;
    const float leftTotalDatapointsToMaintainProbEst = tree.mCounts.Get(leftIndex) / probabilityOfLeft;
    mNumberDatapointsOnNodeCreation[std::make_pair(treeIndex, leftIndex) ] = mNumberDatapointsPerTree[treeIndex] - leftTotalDatapointsToMaintainProbEst;
    mQueuedFrontierLeaves.insert( std::make_pair(treeIndex, leftIndex) );

    const float probabilityOfRightGivenNode = tree.mCounts.Get(rightIndex) / (tree.mCounts.Get(leftIndex) + tree.mCounts.Get(rightIndex));
    const float probabilityOfRight = probabilityOfNode * probabilityOfRightGivenNode;
    const float rightTotalDatapointsToMaintainProbEst = tree.mCounts.Get(rightIndex) / probabilityOfRight;
    mNumberDatapointsOnNodeCreation[std::make_pair(treeIndex, rightIndex) ] = mNumberDatapointsPerTree[treeIndex] - rightTotalDatapointsToMaintainProbEst;
    mQueuedFrontierLeaves.insert( std::make_pair(treeIndex, rightIndex) );
}