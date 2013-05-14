#pragma once

#include <BufferCollection.h>
#include <BufferCollectionStack.h>
#include <Forest.h>
#include <Constants.h>
#include <PipelineStepI.h>

class ForestPredictor
{
public:
    ForestPredictor( const Forest& forest );

    void PredictLeafs(BufferCollection& data, const int numberOfindices, Int32MatrixBuffer& leafsOut);
    void PredictYs(BufferCollection& data, const int numberOfindices, Float32MatrixBuffer& ysOut);
    // void PredictMaxYs(BufferCollection& data, const int numberOfindices, Int32VectorBuffer& maxYsOut);

    Forest mForest;
};

void ForestPredictLeafs(const Forest& forest, BufferCollection& data, const int numberOfindices, Int32MatrixBuffer& leafsOut);
void ForestPredictYs(const Forest& forest, BufferCollection& data, const int numberOfindices, Float32MatrixBuffer& ysOut);
// void ForestPredictMaxYs(const Forest& forest, BufferCollection& data, const int numberOfindices, Int32VectorBuffer& maxYsOut);
int walkTree_old( const Tree& tree, int nodeId, BufferCollection& data, const int index, int& treeDepthOut );


template <class FeatureBinding, class FloatType, class IntType>
IntType nextChild(  const FeatureBinding& feature,
                    const Tree& tree,
                    const IntType nodeId,
                    const IntType index ) 
{
    const FloatType splitpoint = tree.mFloatFeatureParams.Get(nodeId, SPLIT_POINT_INDEX);
    const FloatType featureValue = feature.FeatureValue(nodeId, index);
    bool goLeft = (featureValue > splitpoint);
    const IntType childDirection = goLeft ? 0 : 1;
    const IntType childNodeId = tree.mPath.Get(nodeId, childDirection);
    return childNodeId;
}

template <class FeatureBinding, class FloatType, class IntType>
IntType walkTree( const FeatureBinding& feature,
                  const Tree& tree,
                  const IntType nodeId,
                  const IntType index ) 
{
    const IntType childNodeId = nextChild<FeatureBinding,FloatType,IntType>( feature, tree, nodeId, index);
    if(childNodeId == NULL_CHILD)
    {
       return nodeId;
    }
    return walkTree<FeatureBinding,FloatType,IntType>(feature, tree, childNodeId, index);
}


template <class Feature, class Combiner, class FloatType, class IntType>
class TemplateForestPredictor
{
public:
    TemplateForestPredictor( const Forest& forest, const Feature& feature, const Combiner& combiner, const PipelineStepI* preSteps );
    ~TemplateForestPredictor();

    void PredictLeafs(const BufferCollection& data, MatrixBufferTemplate<IntType>& leafsOut) const;
    void PredictYs(const BufferCollection& data, MatrixBufferTemplate<FloatType>& ysOut);

    Forest GetForest() const;

private:
    const Forest mForest;
    Feature mFeature;
    Combiner mCombiner;
    const PipelineStepI* mPreSteps;
};

template <class Feature, class Combiner, class FloatType, class IntType>
TemplateForestPredictor<Feature, Combiner, FloatType, IntType>::TemplateForestPredictor( const Forest& forest, const Feature& feature, const Combiner& combiner, const PipelineStepI* preSteps )
: mForest(forest)
, mFeature(feature)
, mCombiner(combiner)
, mPreSteps(preSteps->Clone())
{}

template <class Feature, class Combiner, class FloatType, class IntType>
TemplateForestPredictor<Feature, Combiner, FloatType, IntType>::~TemplateForestPredictor()
{
    delete mPreSteps;
}

template <class Feature, class Combiner, class FloatType, class IntType>
void TemplateForestPredictor<Feature, Combiner, FloatType, IntType>::PredictLeafs( const BufferCollection& data,
                                                                                  MatrixBufferTemplate<IntType>& leafsOut) const
{
    boost::mt19937 gen;
    gen.seed(0);

    const int numberOfTreesInForest = mForest.mTrees.size();
    BufferCollectionStack stack;
    stack.Push(&data);

    BufferCollection* perTreeBufferCollection = new BufferCollection[numberOfTreesInForest];
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTreesInForest);
    for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
    {
        BufferCollection& bc = perTreeBufferCollection[treeId];
        bc.AddBuffer< MatrixBufferTemplate<FloatType> >(mFeature.mFloatParamsBufferId, mForest.mTrees[treeId].mFloatFeatureParams);
        bc.AddBuffer< MatrixBufferTemplate<IntType> >(mFeature.mIntParamsBufferId, mForest.mTrees[treeId].mIntFeatureParams);
        mPreSteps->ProcessStep(stack, bc, gen);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    leafsOut.Resize(numberOfIndices, numberOfTreesInForest);

    for(IntType i=0; i<numberOfIndices; i++)
    {
        for(IntType treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            IntType leafNodeId = walkTree<typename Feature::FeatureBinding, FloatType, IntType>(
                                          featureBindings[treeId], mForest.mTrees[treeId], 0, i);
            leafsOut.Set(i, treeId, leafNodeId);
        }
    }

    delete[] perTreeBufferCollection;
}

template <class Feature, class Combiner, class FloatType, class IntType>
void TemplateForestPredictor<Feature, Combiner, FloatType, IntType>::PredictYs( const BufferCollection& data,
                                                                              MatrixBufferTemplate<FloatType>& ysOut)
{
    boost::mt19937 gen;
    gen.seed(0);

    const int numberOfTreesInForest = mForest.mTrees.size();
    BufferCollectionStack stack;
    stack.Push(&data);

    BufferCollection* perTreeBufferCollection = new BufferCollection[numberOfTreesInForest];
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTreesInForest);
    for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
    {
        BufferCollection& bc = perTreeBufferCollection[treeId];
        bc.AddBuffer< MatrixBufferTemplate<FloatType> >(mFeature.mFloatParamsBufferId, mForest.mTrees[treeId].mFloatFeatureParams);
        bc.AddBuffer< MatrixBufferTemplate<IntType> >(mFeature.mIntParamsBufferId, mForest.mTrees[treeId].mIntFeatureParams);
        mPreSteps->ProcessStep(stack, bc, gen);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    ysOut.Resize(numberOfIndices, mCombiner.GetResultDim());

    for(IntType i=0; i<numberOfIndices; i++)
    {
        mCombiner.Reset();
        for(IntType treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            IntType leafNodeId = walkTree<typename Feature::FeatureBinding, FloatType, IntType>(
                                        featureBindings[treeId], mForest.mTrees[treeId], 0, i);
            mCombiner.Combine(leafNodeId, mForest.mTrees[treeId].mYs);
        }
        mCombiner.WriteResult(i, ysOut);
    }

    delete[] perTreeBufferCollection;
}

template <class Feature, class Combiner, class FloatType, class IntType>
Forest TemplateForestPredictor<Feature, Combiner, FloatType, IntType>::GetForest() const
{
    return mForest;
}



