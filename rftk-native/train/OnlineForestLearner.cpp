#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <ctime>
#include <cfloat>
#include <cstdio>

#include "BufferCollection.h"
#include "ForestPredictor.h"
#include "OnlineForestLearner.h"

OnlineForestLearner::OnlineForestLearner( const TrainConfigParams& trainConfigParams )
: mTrainConfigParams(trainConfigParams)
, mForest(  mTrainConfigParams.mNumberOfTrees,
            mTrainConfigParams.mMaxNumberOfNodes,
            mTrainConfigParams.GetIntParamsMaxDim(),
            mTrainConfigParams.GetFloatParamsMaxDim(),
            mTrainConfigParams.GetYDim())
{}

Forest OnlineForestLearner::GetForest() const { return mForest; }

void OnlineForestLearner::Train(BufferCollection data, Int32VectorBuffer indices, OnlineSamplingParams samplingParams )
{
    boost::mt19937 gen( std::time(NULL) );
    boost::poisson_distribution<> poisson(1.0);
    boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(gen, poisson);

    // Loop over trees could be farmed out to different jobs
    for(int treeIndex=0; treeIndex<mForest.mTrees.size(); treeIndex++)
    {
        printf("OnlineForestLearner::Train tree=%d\n", treeIndex);

        Float32VectorBuffer weights(indices.GetN());
        if( samplingParams.mUsePoisson )
        {
            for(int i=0; i<weights.GetN(); i++)
            {
                const float possionValue = static_cast<float>(var_poisson());
                weights.Set(i,possionValue);
            }
        }
        else
        {
            weights.SetAll(1.0f);
        }
        data.AddFloat32VectorBuffer(SAMPLE_WEIGHTS, weights);

        // Iterate over each sample (this cannot be farmed out to different threads)
        Tree& tree = mForest.mTrees[treeIndex];
        for(int sampleIndex=0; sampleIndex<indices.GetN(); sampleIndex++)
        {
            if( weights.Get(sampleIndex) < FLT_EPSILON )
            {
                continue;
            }

            int treeDepth = 0;
            const int nodeIndex = walkTree( tree, 0, data, sampleIndex, treeDepth );

            Int32VectorBuffer singleIndex(1);
            singleIndex.Set(0, indices.Get(sampleIndex));

            std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, nodeIndex);
            if( mActiveNodes.find(treeNodeKey) == mActiveNodes.end() )
            {
                mActiveNodes[treeNodeKey] = new ActiveSplitNode( mTrainConfigParams.mFeatureExtractors,
                                                                mTrainConfigParams.mNodeDataCollectorFactory,
                                                                mTrainConfigParams.mBestSplit,
                                                                mTrainConfigParams.mSplitCriteria,
                                                                treeDepth);
            }

            ActiveSplitNode* activeSplit = mActiveNodes[treeNodeKey];
            activeSplit->ProcessData(data, singleIndex, gen);

            if( activeSplit->ShouldSplit() == SPLT_CRITERIA_READY_TO_SPLIT )
            {
                // printf("OnlineForestLearner::Train SPLIT tree=%d nodeIndex=%d treeDepth=%d\n", treeIndex, nodeIndex, treeDepth);
                tree.mLastNodeIndex++;
                const int leftNode = tree.mLastNodeIndex;
                tree.mLastNodeIndex++;
                const int rightNode = tree.mLastNodeIndex;
                // Todo: Only updating Ys on split, should update Ys everytime
                activeSplit->WriteToTree(nodeIndex, tree.mPath, tree.mFloatFeatureParams, tree.mIntFeatureParams, tree.mDepths,
                                        leftNode, tree.mYs,
                                        rightNode, tree.mYs);

                // Remove split node
                mActiveNodes.erase(treeNodeKey);
                delete activeSplit;
            }
        }
        printf("Tree=%d mLastNodeIndex=%d\n", treeIndex, tree.mLastNodeIndex);
    }
}