#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <ctime>
#include <cfloat>
#include <cstdio>
#include <iostream>

#include "BufferCollection.h"
#include "ForestPredictor.h"
#include "OnlineForestLearner.h"

OnlineForestLearner::OnlineForestLearner( const TrainConfigParams& trainConfigParams,
                                          const OnlineSamplingParams& samplingParams, 
                                          const unsigned int maxFrontierSize
 )
: mTrainConfigParams(trainConfigParams)
, mOnlineSamplingParams(samplingParams)
, mMaxFrontierSize(maxFrontierSize)
, mForest(  mTrainConfigParams.mNumberOfTrees,
            mTrainConfigParams.mMaxNumberOfNodes,
            mTrainConfigParams.GetIntParamsMaxDim(),
            mTrainConfigParams.GetFloatParamsMaxDim(),
            mTrainConfigParams.GetYDim())
, mQueuedFrontierLeaves()
, mActiveFrontierLeaves()
, mNumberOfDatapointsProcessedByTree(mTrainConfigParams.mNumberOfTrees)
, mNumberOfDatapointsProcessedByTreeWhenNodeCreated()
{}

OnlineForestLearner::~OnlineForestLearner()
{
    typedef std::map< std::pair<int, int>, ActiveSplitNode* >::iterator it_type;
    for(it_type iterator = mActiveFrontierLeaves.begin(); iterator != mActiveFrontierLeaves.end(); ++iterator) 
    {
        delete iterator->second;
        iterator->second = NULL;
    }
}

Forest OnlineForestLearner::GetForest() const { return mForest; }

void OnlineForestLearner::Train(BufferCollection data, Int32VectorBuffer indices )
{
    boost::mt19937 gen( std::time(NULL) );
    boost::poisson_distribution<> poisson(1.0);
    boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(gen, poisson);

    // Setup the buffer for the weights
    Float32VectorBuffer weights(indices.GetN());
    data.AddFloat32VectorBuffer(SAMPLE_WEIGHTS, weights);

    for(int sampleIndex=0; sampleIndex<indices.GetN(); sampleIndex++)
    {
        for(unsigned int treeIndex=0; treeIndex<mForest.mTrees.size(); treeIndex++)
        {
            Tree& tree = mForest.mTrees[treeIndex];
            float sampleWeight = 1.0f;
            if( mOnlineSamplingParams.mUsePoisson )
            {
                sampleWeight = static_cast<float>(var_poisson());
            }

            if( sampleWeight < FLT_EPSILON )
            {
                continue;
            }

            mNumberOfDatapointsProcessedByTree[treeIndex]++;

            int treeDepth = 0;
            const int nodeIndex = walkTree( tree, 0, data, sampleIndex, treeDepth );

            const Int32VectorBuffer& classLabels = data.GetInt32VectorBuffer(CLASS_LABELS);
            const int classLabel = classLabels.Get(indices.Get(sampleIndex));

            // Update class histogram (this needs to be extended to support regression)
            const float oldN = tree.mCounts.Get(nodeIndex);
            for(int c=0; c<tree.mYs.GetN(); c++)
            {
                float classCount = oldN * tree.mYs.Get(nodeIndex,c);
                if( classLabel == c )
                {
                    classCount += sampleWeight;
                }
                const float cProbNew = classCount / (oldN + sampleWeight);
                tree.mYs.Set(nodeIndex, c, cProbNew);
            }

            tree.mCounts.Incr(nodeIndex, sampleWeight);

            std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, nodeIndex);
            const bool inQueueFrontier = mQueuedFrontierLeaves.find(treeNodeKey) != mQueuedFrontierLeaves.end();   
            const bool inActiveFrontier = mActiveFrontierLeaves.find(treeNodeKey) != mActiveFrontierLeaves.end();         

            // If leaf is not active or in the queue then add it to the queue
            // This should only happen for the root node because the split logic adds the children to the queu
            if( !inQueueFrontier && !inActiveFrontier )
            {
                mNumberOfDatapointsProcessedByTreeWhenNodeCreated[treeNodeKey] = 0;
                mQueuedFrontierLeaves.insert(treeNodeKey);
            }

            // Add a new active split to the frontier when there is space and there is a candidate in the queue
            // Choose the next candidate with the high probability of error 
            // (this needs to be extended to support regression)
            while( mActiveFrontierLeaves.size() < mMaxFrontierSize && mQueuedFrontierLeaves.size() > 0 )
            {
                // Find node in queue with hprobability of improvement
                float maxProbabilityOfError = 0.0;
                std::pair<int, int> maxProbabilityOfErrorTreeNodeKey = *mQueuedFrontierLeaves.begin();
                typedef std::set< std::pair<int, int> >::iterator it_type;
                for(it_type iter = mQueuedFrontierLeaves.begin(); iter != mQueuedFrontierLeaves.end(); ++iter) 
                {
                    const Tree& nodeTree = mForest.mTrees[iter->first];
                    const long long lifeOfNode = mNumberOfDatapointsProcessedByTree[iter->first] - mNumberOfDatapointsProcessedByTreeWhenNodeCreated[*iter];
                    const float dataPointsToReachNode = nodeTree.mCounts.Get(iter->second);
                    const float probOfNode = dataPointsToReachNode / static_cast<float>(lifeOfNode);  
                    const float* ys = nodeTree.mYs.GetRowPtrUnsafe(iter->second);
                    const float probOfErrorForNode = 1.0f - *std::max_element(ys, ys + nodeTree.mYs.GetN());
                    const float probabilityOfError = probOfNode * probOfErrorForNode;

                    if( probabilityOfError > maxProbabilityOfError )
                    {
                        maxProbabilityOfError = probabilityOfError;
                        maxProbabilityOfErrorTreeNodeKey = *iter;
                    }
                }

                const int maxProbabilityNodeDepth = mForest.mTrees[maxProbabilityOfErrorTreeNodeKey.first].mDepths.Get(maxProbabilityOfErrorTreeNodeKey.second);
                mQueuedFrontierLeaves.erase(maxProbabilityOfErrorTreeNodeKey);
                mActiveFrontierLeaves[maxProbabilityOfErrorTreeNodeKey] = new ActiveSplitNode( mTrainConfigParams.mFeatureExtractors,
                                                                mTrainConfigParams.mNodeDataCollectorFactory,
                                                                mTrainConfigParams.mBestSplit,
                                                                mTrainConfigParams.mSplitCriteria,
                                                                maxProbabilityNodeDepth,
                                                                mOnlineSamplingParams.mEvalSplitPeriod);
            }

            // Update split
            if( mActiveFrontierLeaves.find(treeNodeKey) != mActiveFrontierLeaves.end() )
            {
                ActiveSplitNode* activeSplit = mActiveFrontierLeaves[treeNodeKey];
                Int32VectorBuffer singleIndex(1);
                singleIndex.Set(0, indices.Get(sampleIndex));

                Float32VectorBuffer& weights = data.GetFloat32VectorBuffer(SAMPLE_WEIGHTS);
                weights.Set(sampleIndex, sampleWeight);
                activeSplit->ProcessData(data, singleIndex, gen);

                // Split the node
                if( activeSplit->ShouldSplit() == SPLT_CRITERIA_READY_TO_SPLIT )
                {
                    const int leftNode = tree.mLastNodeIndex;
                    tree.mLastNodeIndex++;
                    const int rightNode = tree.mLastNodeIndex;
                    tree.mLastNodeIndex++;
                    activeSplit->WriteToTree(nodeIndex, leftNode, rightNode,
                                            tree.mPath, tree.mFloatFeatureParams, tree.mIntFeatureParams,
                                            tree.mDepths, tree.mCounts, tree.mYs);

                    // Remove split node
                    mActiveFrontierLeaves.erase(treeNodeKey);
                    delete activeSplit;

                    // Compute the number of datapoints when the parent started estimating the split
                    // Because the two streams can 'leak' points the actual start time is estimated
                    // from the parents probability of receiving a datapoint and the condition probability
                    // of the datapoint going left or right.  Hopefully this can be simplified
                    const float probOfParent = tree.mCounts.Get(nodeIndex) / static_cast<float>(mNumberOfDatapointsProcessedByTree[treeIndex] - mNumberOfDatapointsProcessedByTreeWhenNodeCreated[treeNodeKey]);
                    const float probOfLeft = tree.mCounts.Get(leftNode) / (tree.mCounts.Get(leftNode) + tree.mCounts.Get(rightNode));
                    const float leftTotalDatapointsToMaintainProbEst = tree.mCounts.Get(leftNode) / (probOfParent * probOfLeft);
                    mNumberOfDatapointsProcessedByTreeWhenNodeCreated[std::make_pair(treeIndex, leftNode) ] = mNumberOfDatapointsProcessedByTree[treeIndex] - leftTotalDatapointsToMaintainProbEst;
                    mQueuedFrontierLeaves.insert( std::make_pair(treeIndex, leftNode) );

                    const float probOfRight = tree.mCounts.Get(rightNode) / (tree.mCounts.Get(leftNode) + tree.mCounts.Get(rightNode));
                    const float rightTotalDatapointsToMaintainProbEst = tree.mCounts.Get(rightNode) / (probOfParent * probOfRight);
                    mNumberOfDatapointsProcessedByTreeWhenNodeCreated[std::make_pair(treeIndex, rightNode) ] = mNumberOfDatapointsProcessedByTree[treeIndex] - rightTotalDatapointsToMaintainProbEst;
                    mQueuedFrontierLeaves.insert( std::make_pair(treeIndex, rightNode) );
                }
            }
        }
    }
}