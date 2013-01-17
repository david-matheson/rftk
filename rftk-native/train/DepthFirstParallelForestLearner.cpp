#include <stdio.h>
#include <vector>

#if USE_BOOST_THREAD
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#endif

#include "bootstrap.h"
#include "assert_util.h"

#include "OfflineSamplingParams.h"
#include "DepthFirstParallelForestLearner.h"

void TrainTrees(   BufferCollection& data,
                    MatrixBufferInt& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    int startIndex,
                    int offset,
                    Forest* forestOut );

void TrainTree(    BufferCollection& data,
                    MatrixBufferInt& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    Tree* treeOut);


void ProcessNode(   const TrainConfigParams& trainConfigParams,
                    int nodeIndex,
                    int treeDepth,
                    BufferCollection& data,
                    const MatrixBufferInt& indices,
                    Tree* treeOut);

DepthFirstParallelForestLearner::DepthFirstParallelForestLearner( const TrainConfigParams& trainConfigParams )
: mTrainConfigParams(trainConfigParams)
{}

Forest DepthFirstParallelForestLearner::Train(  BufferCollection data,
                                                MatrixBufferInt indices,
                                                const OfflineSamplingParams& samplingParams,
                                                int numberOfJobs )
{
#if USE_BOOST_THREAD
    // Putting forest on heap because different threads are going to write to it
    Forest* forest = new Forest(mTrainConfigParams.mNumberOfTrees);
    std::vector< boost::shared_ptr< boost::thread > > threadVec;
    for(int job=0; job<numberOfJobs; job++)
    {
        threadVec.push_back(boost::make_shared<boost::thread>(TrainTrees, data, indices, mTrainConfigParams, samplingParams, job, numberOfJobs, forest));
    }
    for(int job=0; job<numberOfJobs; job++)
    {
        threadVec[job]->join();
    }
    Forest result = *forest;
    delete forest;
#else
    Forest result(mTrainConfigParams.mNumberOfTrees);
    TrainTrees(data, indices, mTrainConfigParams, samplingParams, 0, 1, &result);
#endif
    return result;
}

void TrainTrees(   BufferCollection& data,
                    MatrixBufferInt& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    int startIndex,
                    int offset,
                    Forest* forestOut )
{
    // printf("DepthFirstParallelForestLearner::Train startIndex=%d offest=%d\n", startIndex, offset);
    for(int i=startIndex; i<trainConfigParams.mNumberOfTrees; i+=offset)
    {
        // printf("DepthFirstParallelForestLearner::Train tree=%d\n", i);
        forestOut->mTrees[i] = Tree(    trainConfigParams.mMaxNumberOfNodes,
                                        trainConfigParams.GetIntParamsMaxDim(),
                                        trainConfigParams.GetFloatParamsMaxDim(),
                                        trainConfigParams.GetYDim());
        TrainTree(data, indices, trainConfigParams, samplingParams, &forestOut->mTrees[i]);
    }
}


void TrainTree(    BufferCollection& data,
                    MatrixBufferInt& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    Tree* treeOut)
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
    ProcessNode(trainConfigParams, 0, 0, data, sampledIndicesBuffer, treeOut);
}


void ProcessNode(   const TrainConfigParams& trainConfigParams,
                    int nodeIndex,
                    int treeDepth,
                    BufferCollection& data,
                    const MatrixBufferInt& indices,
                    Tree* treeOut)
{
    // printf("DepthFirstParallelForestLearner::ProcessNode depth=%d\n", treeDepth);
    ActiveSplitNode activeSplit = ActiveSplitNode(  trainConfigParams.mFeatureExtractors,
                                                    trainConfigParams.mNodeDataCollectorFactory,
                                                    trainConfigParams.mBestSplit,
                                                    trainConfigParams.mSplitCriteria,
                                                    treeDepth);

    activeSplit.ProcessData(data, indices);
    if( activeSplit.ShouldSplit() == SPLT_CRITERIA_READY_TO_SPLIT )
    {
        treeOut->mLastNodeIndex++;
        const int leftNode = treeOut->mLastNodeIndex;
        treeOut->mLastNodeIndex++;
        const int rightNode = treeOut->mLastNodeIndex;
        activeSplit.WriteToTree(nodeIndex, treeOut->mPath, treeOut->mFloatFeatureParams, treeOut->mIntFeatureParams, treeOut->mDepths,
                                leftNode, treeOut->mYs,
                                rightNode, treeOut->mYs);

        MatrixBufferInt leftIndices;
        MatrixBufferInt rightIndices;
        activeSplit.SplitIndices(data, indices, leftIndices, rightIndices);

        ProcessNode(trainConfigParams, leftNode, treeDepth+1, data, leftIndices, treeOut);
        ProcessNode(trainConfigParams, rightNode, treeDepth+1, data, rightIndices, treeOut);
    }
}