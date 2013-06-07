#pragma once

#include "DepthFirstTreeLearner.h"

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"

#include "TrySplitCriteriaI.h"

#include "PipelineStepI.h"
#include "MinNodeSizeCriteria.h"
#include "AllSamplesStep.h"
#include "SliceBufferStep.h"
#include "SetBufferStep.h"
#include "AxisAlignedParamsStep.h"
#include "LinearMatrixFeature.h"
#include "FeatureExtractorStep.h"
#include "ClassInfoGainWalker.h"
#include "BestSplitpointsWalkingSortedStep.h"
#include "Pipeline.h"

#include "MinImpurityCriteria.h"
#include "ClassEstimatorFinalizer.h"
#include "SplitSelector.h"

#include "DepthFirstTreeLearner.h"
#include "Tree.h"

template<typename T>
VectorBufferTemplate<T> CreateVector1(T value)
{
    T data[] = {value};
    return VectorBufferTemplate<T>(&data[0], 1);
}

template<typename T>
MatrixBufferTemplate<T> CreateXsMatrix10x2()
{
    T data[] = {5,-3,
                5,-3,
                5,2,
                5,2,
                5,0,
                -1,-3,
                -1,-3,
                -1,2,
                -1,2,
                -1,0};

    return MatrixBufferTemplate<T>(&data[0], 10, 2);
}

template<typename T>
VectorBufferTemplate<T> CreateClassesVector10()
{
    T data[] = {0,0,0,0,0,1,1,2,2,3};
    return VectorBufferTemplate<T>(&data[0], 10);
}

struct DepthFirstTreeLearnerFixture {
    DepthFirstTreeLearnerFixture()
    : xs_key("xs")
    , xs( CreateXsMatrix10x2<float>() )
    , classes_key("classes")
    , classes( CreateClassesVector10<int>() )
    , collection()
    {
        collection.AddBuffer(xs_key, xs);
        collection.AddBuffer(classes_key, classes);
        stack.Push(&collection);
    }

    ~DepthFirstTreeLearnerFixture()
    {
    }

    const BufferCollectionKey_t xs_key;
    const MatrixBufferTemplate<float> xs;
    const BufferCollectionKey_t classes_key;
    const VectorBufferTemplate<int> classes;    
    BufferCollection collection;
    BufferCollectionStack stack;
};

DepthFirstTreeLearner<float, int> CreateDepthFirstLearner( BufferCollectionKey_t xs_key, 
                                                          BufferCollectionKey_t classes_key, 
                                                          int numberOfClasses, 
                                                          FeatureValueOrdering featureOrdering, 
                                                          double minNodeSize);