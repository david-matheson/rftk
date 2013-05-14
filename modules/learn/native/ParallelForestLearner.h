#pragma once

#include "BufferCollectionStack.h"
#include "TreeLearnerI.h"
#include "Forest.h"

class ParallelForestLearner
{
public:
    ParallelForestLearner( const TreeLearnerI* treeLearner, 
                            int numberOfTrees, 
                            int maxIntParamsDim, 
                            int maxFloatParamsDim, 
                            int maxEstimatorParamsDim, 
                            int numberOfJobs );
    ~ParallelForestLearner();

    Forest Learn( const BufferCollection& data );
private:
    ParallelForestLearner( const ParallelForestLearner& other );
    ParallelForestLearner& operator=( const ParallelForestLearner& rhs );
    
    Forest* mForest;  // because multiple threads write to mForest, it must be on the heap
    const TreeLearnerI* mTreeLearner;
    const int mNumberOfTrees;
    const int mNumberOfJobs;
};

