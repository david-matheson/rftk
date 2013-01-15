#include "DepthFirstParallelForestLearner.h"


DepthFirstParallelForestLearner::DepthFirstParallelForestLearner( const TrainConfigParams& trainConfigParams )
: mTrainConfigParams(trainConfigParams)
{}

Forest DepthFirstParallelForestLearner::Train(  BufferCollection data,
                                                MatrixBufferInt indices,
                                                const OfflineSamplingParams& samplingParams,
                                                int numberOfJobs )
{
    Forest forest(mTrainConfigParams.mNumberOfTrees);
    //Todo use a pool of threads for training each tree
    for(int i=0; i<mTrainConfigParams.mNumberOfTrees; i++)
    {
        forest.mTrees[i] = Tree(    mTrainConfigParams.mMaxNumberOfNodes,
                                    mTrainConfigParams.GetIntParamsMaxDim(),
                                    mTrainConfigParams.GetFloatParamsMaxDim(),
                                    mTrainConfigParams.GetYDim());
        TrainTree(data, indices, samplingParams, forest.mTrees[i]);
    }
    return forest;
}


void DepthFirstParallelForestLearner::TrainTree(    BufferCollection data,
                                                    MatrixBufferInt indices,
                                                    const OfflineSamplingParams& samplingParams,
                                                    Tree& treeOut)
{
    // Sample weights
    // Count weights > 0
    MatrixBufferInt indicesGreaterThanZero;
    // Fill in indicesGreaterThanZero > 0
    ProcessNode(0, 0, data, indicesGreaterThanZero, treeOut);
}


void DepthFirstParallelForestLearner::ProcessNode(  int nodeIndex,
                                                    int treeDepth,
                                                    BufferCollection& data,
                                                    const MatrixBufferInt& indices,
                                                    Tree& treeOut)
{
    ActiveSplitNode activeSplit = ActiveSplitNode(  mTrainConfigParams.mFeatureExtractors,
                                                    mTrainConfigParams.mNodeDataCollectorFactory,
                                                    mTrainConfigParams.mBestSplit,
                                                    mTrainConfigParams.mSplitCriteria,
                                                    treeDepth);

    activeSplit.ProcessData(data, indices);
    if( activeSplit.ShouldSplit() == SPLT_CRITERIA_READY_TO_SPLIT )
    {
        treeOut.mLastNodeIndex++;
        const int leftNode = treeOut.mLastNodeIndex;
        treeOut.mLastNodeIndex++;
        const int rightNode = treeOut.mLastNodeIndex;
        activeSplit.WriteToTree(nodeIndex, treeOut.mPath, treeOut.mFloatFeatureParams, treeOut.mIntFeatureParams, treeOut.mDepths,
                                leftNode, treeOut.mYs,
                                rightNode, treeOut.mYs);

        MatrixBufferInt leftIndices;
        MatrixBufferInt rightIndices;
        activeSplit.SplitIndices(data, indices, leftIndices, rightIndices);

        ProcessNode(leftNode, treeDepth+1, data, leftIndices, treeOut);
        ProcessNode(rightNode, treeDepth+1, data, rightIndices, treeOut);
    }
}