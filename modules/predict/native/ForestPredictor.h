#pragma once

#include <BufferCollection.h>
#include <BufferCollectionStack.h>
#include <Forest.h>
#include <Constants.h>
#include <PipelineStepI.h>

template <class FeatureBinding, class BufferTypes>
typename BufferTypes::Index nextChild(  const FeatureBinding& feature,
                    const Tree& tree,
                    const typename BufferTypes::Index nodeId,
                    const typename BufferTypes::Index index )
{
    const typename BufferTypes::FeatureValue splitpoint = tree.mFloatFeatureParams.Get(nodeId, SPLIT_POINT_INDEX);
    const typename BufferTypes::FeatureValue featureValue = feature.FeatureValue(nodeId, index);
    bool goLeft = (featureValue > splitpoint);
    const typename BufferTypes::Index childDirection = goLeft ? 0 : 1;
    const typename BufferTypes::Index childNodeId = tree.mPath.Get(nodeId, childDirection);
    return childNodeId;
}

template <class FeatureBinding, class BufferTypes>
typename BufferTypes::Index walkTree( const FeatureBinding& feature,
                  const Tree& tree,
                  const typename BufferTypes::Index nodeId,
                  const typename BufferTypes::Index index )
{
    const typename BufferTypes::Index childNodeId = nextChild<FeatureBinding,BufferTypes>( feature, tree, nodeId, index);
    if(childNodeId == NULL_CHILD)
    {
       return nodeId;
    }
    return walkTree<FeatureBinding,BufferTypes>(feature, tree, childNodeId, index);
}


template <class Feature, class Combiner, class BufferTypes>
class TemplateForestPredictor
{
public:
    TemplateForestPredictor( const Forest& forest, const Feature& feature, const Combiner& combiner, const PipelineStepI* preSteps );
    ~TemplateForestPredictor();

    void PredictLeafs(const BufferCollection& data, MatrixBufferTemplate<int>& leafsOut) const;
    void PredictYs(const BufferCollection& data, MatrixBufferTemplate<float>& ysOut);

    Forest GetForest() const;

private:
    const Forest mForest;
    Feature mFeature;
    Combiner mCombiner;
    const PipelineStepI* mPreSteps;
};

template <class Feature, class Combiner, class BufferTypes>
TemplateForestPredictor<Feature, Combiner, BufferTypes>::TemplateForestPredictor( const Forest& forest, const Feature& feature, const Combiner& combiner, const PipelineStepI* preSteps )
: mForest(forest)
, mFeature(feature)
, mCombiner(combiner)
, mPreSteps(preSteps->Clone())
{}

template <class Feature, class Combiner, class BufferTypes>
TemplateForestPredictor<Feature, Combiner, BufferTypes>::~TemplateForestPredictor()
{
    delete mPreSteps;
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictLeafs( const BufferCollection& data,
                                                                                  MatrixBufferTemplate<int>& leafsOut) const
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
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFeature.mFloatParamsBufferId, mForest.mTrees[treeId].mFloatFeatureParams);
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mFeature.mIntParamsBufferId, mForest.mTrees[treeId].mIntFeatureParams);
        mPreSteps->ProcessStep(stack, bc, gen, bc, 0);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    leafsOut.Resize(numberOfIndices, numberOfTreesInForest);

    for(typename BufferTypes::Index i=0; i<numberOfIndices; i++)
    {
        for(typename BufferTypes::Index treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            typename BufferTypes::Index leafNodeId = walkTree<typename Feature::FeatureBinding, BufferTypes>(
                                                                            featureBindings[treeId], mForest.mTrees[treeId], 0, i);
            leafsOut.Set(i, treeId, leafNodeId);
        }
    }

    delete[] perTreeBufferCollection;
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictYs( const BufferCollection& data,
                                                                              MatrixBufferTemplate<float>& ysOut)
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
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFeature.mFloatParamsBufferId, mForest.mTrees[treeId].mFloatFeatureParams);
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mFeature.mIntParamsBufferId, mForest.mTrees[treeId].mIntFeatureParams);
        mPreSteps->ProcessStep(stack, bc, gen, bc, 0);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    ysOut.Resize(numberOfIndices, mCombiner.GetResultDim());

    for(typename BufferTypes::Index i=0; i<numberOfIndices; i++)
    {
        mCombiner.Reset();
        for(typename BufferTypes::Index treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            const Tree& tree = mForest.mTrees[treeId];
            typename BufferTypes::Index leafNodeId = walkTree<typename Feature::FeatureBinding, BufferTypes>(
                                                                featureBindings[treeId], tree, 0, i);
            mCombiner.Combine(leafNodeId, tree.mCounts.Get(leafNodeId), tree.mYs);
        }
        mCombiner.WriteResult(i, ysOut);
    }

    delete[] perTreeBufferCollection;
}

template <class Feature, class Combiner, class BufferTypes>
Forest TemplateForestPredictor<Feature, Combiner, BufferTypes>::GetForest() const
{
    return mForest;
}



