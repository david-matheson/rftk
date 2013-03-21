#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <ctime>
#include <stdio.h>
#include <vector>

#define USE_BOOST_THREAD 0

#if USE_BOOST_THREAD
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#endif

#include "asserts/asserts.h"
#include "bootstrap/bootstrap.h"

#include "OfflineSamplingParams.h"
#include "DepthFirstParallelForestLearner.h"

void TrainTrees(    BufferCollection& data,
                    const Int32VectorBuffer& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    int startIndex,
                    int offset,
                    Forest* forestOut );

void TrainTree(     BufferCollection& data,
                    const Int32VectorBuffer& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    Tree* treeOut);


void ProcessNode(   const TrainConfigParams& trainConfigParams,
                    int nodeIndex,
                    int treeDepth,
                    const BufferCollection& data,
                    const Int32VectorBuffer& indices,
                    boost::mt19937& gen,
                    Tree* treeOut);

DepthFirstParallelForestLearner::DepthFirstParallelForestLearner( const TrainConfigParams& trainConfigParams )
: mTrainConfigParams(trainConfigParams)
{}

Forest DepthFirstParallelForestLearner::Train(  BufferCollection& data,
                                                const Int32VectorBuffer& indices,
                                                const OfflineSamplingParams& samplingParams,
                                                int numberOfJobs )
{
    // Putting forest on heap because different threads are going to write to it
    Forest* forest = new Forest(mTrainConfigParams.mNumberOfTrees);
#if USE_BOOST_THREAD
    std::vector< boost::shared_ptr< boost::thread > > threadVec;
    for(int job=0; job<numberOfJobs; job++)
    {
        threadVec.push_back(boost::make_shared<boost::thread>(TrainTrees, data, indices, mTrainConfigParams, samplingParams, job, numberOfJobs, forest));
    }
    for(int job=0; job<numberOfJobs; job++)
    {
        threadVec[job]->join();
    }
#else
    UNUSED_PARAM(numberOfJobs) // Suppress the unused warning because numberOfJobs is used in USE_BOOST_THREAD
    TrainTrees(data, indices, mTrainConfigParams, samplingParams, 0, 1, forest);
#endif
    Forest result = *forest;
    delete forest;
    return result;
}

void TrainTrees(    BufferCollection& data,
                    const Int32VectorBuffer& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    int startIndex,
                    int offset,
                    Forest* forestOut )
{
    for(int i=startIndex; i<trainConfigParams.mNumberOfTrees; i+=offset)
    {
        // printf("DepthFirstParallelForestLearner::Train tree=%d\n", i);
        forestOut->mTrees[i] = Tree(    trainConfigParams.mInitialNumberOfNodes,
                                        trainConfigParams.GetIntParamsMaxDim(),
                                        trainConfigParams.GetFloatParamsMaxDim(),
                                        trainConfigParams.GetYDim());
    }

    // printf("DepthFirstParallelForestLearner::Train startIndex=%d offest=%d\n", startIndex, offset);
    for(int i=startIndex; i<trainConfigParams.mNumberOfTrees; i+=offset)
    {
        TrainTree(data, indices, trainConfigParams, samplingParams, &forestOut->mTrees[i]);
    }
}


void TrainTree(     BufferCollection& data,
                    const Int32VectorBuffer& indices,
                    const TrainConfigParams& trainConfigParams,
                    const OfflineSamplingParams& samplingParams,
                    Tree* treeOut)
{
    ASSERT_VALID_RANGE(samplingParams.mNumberOfSamples-1, 0, indices.GetN())
    std::vector<int> counts(indices.GetN());
    sample( &counts[0],
            indices.GetN(),
            samplingParams.mNumberOfSamples,
            samplingParams.mWithReplacement);

    Float32VectorBuffer weights(samplingParams.mNumberOfSamples);
    std::vector<int> sampledIndices;
    for(unsigned int i=0; i<counts.size(); i++)
    {
        if( counts[i] > 0 )
        {
            weights.Set(i,static_cast<float>(counts[i]));
            sampledIndices.push_back(i);
        }
    }
    data.AddFloat32VectorBuffer(SAMPLE_WEIGHTS, weights);
    Int32VectorBuffer sampledIndicesBuffer(&sampledIndices[0], sampledIndices.size());
    boost::mt19937 gen( std::time(NULL) );
    ProcessNode(trainConfigParams, 0, 0, data, sampledIndicesBuffer, gen, treeOut);
}


void ProcessNode(   const TrainConfigParams& trainConfigParams,
                    int nodeIndex,
                    int treeDepth,
                    const BufferCollection& data,
                    const Int32VectorBuffer& indices,
                    boost::mt19937& gen,
                    Tree* treeOut)
{
    // printf("DepthFirstParallelForestLearner::ProcessNode depth=%d\n", treeDepth);
    const int evalSplitPeriod = 1;
    ActiveSplitNode* activeSplit = new ActiveSplitNode( trainConfigParams.mFeatureExtractors,
                                                        trainConfigParams.mNodeDataCollectorFactory,
                                                        trainConfigParams.mBestSplit,
                                                        trainConfigParams.mSplitCriteria,
                                                        treeDepth,
                                                        evalSplitPeriod);

    activeSplit->ProcessData(data, indices, gen);
    if( activeSplit->ShouldSplit() == SPLT_CRITERIA_READY_TO_SPLIT )
    {
        const int leftNode = treeOut->NextNodeIndex();
        const int rightNode = treeOut->NextNodeIndex();
        activeSplit->WriteToTree(nodeIndex, leftNode, rightNode,
                                treeOut->mPath, treeOut->mFloatFeatureParams, treeOut->mIntFeatureParams,
                                treeOut->mDepths, treeOut->mCounts, treeOut->mYs);

        Int32VectorBuffer leftIndices;
        Int32VectorBuffer rightIndices;
        activeSplit->SplitIndices(data, indices, leftIndices, rightIndices);

        // Delete before recursing to save memory
        delete activeSplit;
        activeSplit = NULL;

        ProcessNode(trainConfigParams, leftNode, treeDepth+1, data, leftIndices, gen, treeOut);
        ProcessNode(trainConfigParams, rightNode, treeDepth+1, data, rightIndices, gen, treeOut);
    }
    if( activeSplit != NULL )
    {
        delete activeSplit;
    }
}