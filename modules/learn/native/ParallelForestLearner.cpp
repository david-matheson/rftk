#include "ParallelForestLearner.h"

#include "BufferCollectionStack.h"
#include "TreeLearnerI.h"
#include "Forest.h"

#if USE_BOOST_THREAD
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#endif

void TrainTrees(    const TreeLearnerI* treeLearner,
                    const BufferCollection& data,
                    int treeStartIndex,
                    int treeStride,
                    int numberOfTrees,
                    Forest* forestOut )
{
    for(int i=treeStartIndex; i<numberOfTrees; i+=treeStride)
    {
        treeLearner->Learn(data, forestOut->mTrees[i]);
    }
}

ParallelForestLearner::ParallelForestLearner( const TreeLearnerI* treeLearner, int numberOfTrees, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim, int numberOfJobs )
: mForest( new Forest(numberOfTrees, 1, maxIntParamsDim, maxFloatParamsDim, maxYsDim) )
, mTreeLearner( treeLearner->Clone() )
, mNumberOfTrees(numberOfTrees)
, mNumberOfJobs(numberOfJobs)
{}

ParallelForestLearner::~ParallelForestLearner()
{
    delete mForest;
    delete mTreeLearner;
}

Forest& ParallelForestLearner::Learn( const BufferCollection& data )
{
#if USE_BOOST_THREAD
    std::vector< boost::shared_ptr< boost::thread > > threadVec;
    for(int job=0; job<mNumberOfJobs; job++)
    {
        threadVec.push_back( boost::make_shared<boost::thread>(TrainTrees, mTreeLearner, data, job, mNumberOfJobs, mNumberOfTrees, mForest) );
    }
    for(int job=0; job<mNumberOfJobs; job++)
    {
        threadVec[job]->join();
    }
#else
    TrainTrees(mTreeLearner, data, 0, 1, mNumberOfTrees, mForest);
#endif
    return *mForest;
}


