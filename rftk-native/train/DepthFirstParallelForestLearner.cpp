#include "bootstrap.h"
#include "assert_util.h"

#include "OfflineSamplingParams.h"
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
        // printf("DepthFirstParallelForestLearner::Train tree=%d\n", i);
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
    ASSERT_VALID_RANGE(samplingParams.mNumberOfSamples-1, 0, indices.GetM())
    std::vector<int> counts(indices.GetM());
    sample( &counts[0],
            indices.GetM(),
            samplingParams.mNumberOfSamples, 
            samplingParams.mWithReplacement);

    MatrixBufferFloat weights(samplingParams.mNumberOfSamples, 1);
    std::vector<int> sampledIndices;
    for(int i=0; i<counts.size(); i++)
    {
        if( counts[i] > 0 )
        {
            weights.Set(i,0,static_cast<float>(counts[i]));
            sampledIndices.push_back(i);
        }
    }
    data.AddMatrixBufferFloat(SAMPLE_WEIGHTS, weights);
    MatrixBufferInt sampledIndicesBuffer(&sampledIndices[0], sampledIndices.size(), 1);
    ProcessNode(0, 0, data, sampledIndicesBuffer, treeOut);
}


void DepthFirstParallelForestLearner::ProcessNode(  int nodeIndex,
                                                    int treeDepth,
                                                    BufferCollection& data,
                                                    const MatrixBufferInt& indices,
                                                    Tree& treeOut)
{
    // printf("DepthFirstParallelForestLearner::ProcessNode depth=%d\n", treeDepth);
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