#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <ctime>
#include <cfloat>
#include <cstdio>
#include <iostream>

#include <BufferCollection.h>
#include <ForestPredictor.h>

#include "OnlineForestLearner.h"

OnlineForestLearner::OnlineForestLearner( const TrainConfigParams& trainConfigParams,
                                          const OnlineSamplingParams& samplingParams,
                                          const unsigned int maxFrontierSize,
                                          const int maxDepth
 )
: mTrainConfigParams(trainConfigParams)
, mOnlineSamplingParams(samplingParams)
, mMaxFrontierSize(maxFrontierSize)
, mMaxDepth(maxDepth)
, mForest(  mTrainConfigParams.mNumberOfTrees,
            mTrainConfigParams.mInitialNumberOfNodes,
            mTrainConfigParams.GetIntParamsMaxDim(),
            mTrainConfigParams.GetFloatParamsMaxDim(),
            mTrainConfigParams.GetYDim())
, mFrontierQueue(mTrainConfigParams.mNumberOfTrees)
, mActiveFrontierLeaves()
{
    UpdateActiveFrontier();
}

OnlineForestLearner::~OnlineForestLearner()
{
    typedef std::map< std::pair<int, int>, ActiveSplitNode* >::iterator it_type;
    for(it_type iterator = mActiveFrontierLeaves.begin(); iterator != mActiveFrontierLeaves.end(); ++iterator)
    {
        delete iterator->second;
        iterator->second = NULL;
    }
}

Forest OnlineForestLearner::GetForest() const
{
    return mForest;
}

void OnlineForestLearner::Train(BufferCollection data, Int32VectorBuffer indices )
{
    boost::mt19937 gen( std::time(NULL) );
    boost::poisson_distribution<> poisson(1.0);
    boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(gen, poisson);

    // Setup the buffer for the weights
    Float32VectorBuffer weights(indices.GetMax()+1);
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


            mFrontierQueue.IncrDatapoints(treeIndex, static_cast<long long>(sampleWeight));

            int treeDepth = 0;
            const int nodeIndex = walkTree_old( tree, 0, data, indices.Get(sampleIndex), treeDepth );

            // Update class histogram (this needs to be moved to a seperate class to support regression)
            const Int32VectorBuffer& classLabels = data.GetInt32VectorBuffer(CLASS_LABELS);
            const int classLabel = classLabels.Get(indices.Get(sampleIndex));
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

            // Update active node stats
            std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, nodeIndex);
            if( mActiveFrontierLeaves.find(treeNodeKey) != mActiveFrontierLeaves.end() )
            {
                ActiveSplitNode* activeSplit = mActiveFrontierLeaves[treeNodeKey];
                Int32VectorBuffer singleIndex(1);
                singleIndex.Set(0, indices.Get(sampleIndex));

                Float32VectorBuffer& weights = data.GetFloat32VectorBuffer(SAMPLE_WEIGHTS);
                weights.Set(indices.Get(sampleIndex), sampleWeight);
                activeSplit->ProcessData(data, singleIndex, gen);

                // Split the node
                if( activeSplit->ShouldSplit() == SPLT_CRITERIA_READY_TO_SPLIT )
                {
                    const int leftNode = tree.NextNodeIndex();
                    const int rightNode = tree.NextNodeIndex();
                    activeSplit->WriteToTree(nodeIndex, leftNode, rightNode,
                                            tree.mPath, tree.mFloatFeatureParams, tree.mIntFeatureParams,
                                            tree.mDepths, tree.mCounts, tree.mYs);

                    // Remove split node
                    mActiveFrontierLeaves.erase(treeNodeKey);
                    delete activeSplit;

                    // Only add nodes to the priority queue if they're less than the max depth
                    if( treeDepth+1 < mMaxDepth )
                    {
                        mFrontierQueue.ProcessSplit(mForest, treeIndex, nodeIndex, leftNode, rightNode);
                    }

                    // Update the frontier priority queue
                    UpdateActiveFrontier();
                }
            }
        }
    }
}

void OnlineForestLearner::UpdateActiveFrontier()
{
    // Add a new active split to the frontier when there is space and there is a candidate in the queue
    while( mActiveFrontierLeaves.size() < mMaxFrontierSize && !mFrontierQueue.IsEmpty() )
    {
        std::pair<int,int> treeNodeKey = mFrontierQueue.PopBest(mForest);
        const int treeNodeKeyDepth = mForest.mTrees[treeNodeKey.first].mDepths.Get(treeNodeKey.second);
        mActiveFrontierLeaves[treeNodeKey] = new ActiveSplitNode( mTrainConfigParams.mFeatureExtractors,
                                                        mTrainConfigParams.mNodeDataCollectorFactory,
                                                        mTrainConfigParams.mBestSplit,
                                                        mTrainConfigParams.mSplitCriteria,
                                                        treeNodeKeyDepth,
                                                        mOnlineSamplingParams.mEvalSplitPeriod);
    }
}