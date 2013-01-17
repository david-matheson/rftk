#include "BufferCollection.h"
#include "VecPredict.h"
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

void OnlineForestLearner::Train(BufferCollection data, MatrixBufferInt indices )
{
    VecForestPredictor forestPredictor = VecForestPredictor(mForest);
    MatrixBufferInt leafs;
    forestPredictor.PredictLeafs(data.GetMatrixBufferFloat(X_FLOAT_DATA), leafs);

    // Loop over trees could be farmed out to different jobs
    for(int treeIndex=0; treeIndex<mForest.mTrees.size(); treeIndex++)
    {
        printf("OnlineForestLearner::Train tree=%d\n", treeIndex);
        // Add weights to data
        // MatrixBufferFloat singleWeight(1,0);
        //singleWeight ~ Possion
        MatrixBufferFloat weights(indices.GetM(),1);
        weights.SetAll(1.0);
        data.AddMatrixBufferFloat(SAMPLE_WEIGHTS, weights);

        // Iterate over each sample (this cannot be farmed out to different threads)
        for(int sampleIndex=0; sampleIndex<indices.GetM(); sampleIndex++)
        {
            Tree& tree = mForest.mTrees[treeIndex];
            int nodeIndex = leafs.Get(sampleIndex, treeIndex);
            int treeDepth = tree.mDepths.Get(nodeIndex,0);

            MatrixBufferInt singleIndex(1,1);
            singleIndex.Set(0,0, indices.Get(sampleIndex, 0));

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
            activeSplit->ProcessData(data, singleIndex);

            if( activeSplit->ShouldSplit() == SPLT_CRITERIA_READY_TO_SPLIT )
            {
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
    }
}