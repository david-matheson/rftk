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
    void PredictOobYs(const BufferCollection& data, MatrixBufferTemplate<float>& ysOut);

    Forest GetForest() const;

private:
    void PredictYsInternal(const BufferCollection& data, MatrixBufferTemplate<float>& ysOut, bool useOobIndices);

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
    PredictYsInternal(data, ysOut, false);
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictOobYs( const BufferCollection& data,
                                                                              MatrixBufferTemplate<float>& ysOut)
{
    PredictYsInternal(data, ysOut, true);
}



template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictYsInternal( const BufferCollection& data,
                                                                              MatrixBufferTemplate<float>& ysOut,
                                                                              bool useOobIndices)
{
    boost::mt19937 gen;
    gen.seed(0);

    const int numberOfTreesInForest = mForest.mTrees.size();
    BufferCollectionStack stack;
    stack.Push(&data);

    std::vector<BufferCollection> perTreeBufferCollection(numberOfTreesInForest);
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTreesInForest);

    std::vector< const VectorBufferTemplate<typename BufferTypes::Index>* > oobIndices(numberOfTreesInForest);
    std::vector<typename BufferTypes::Index> currentOobOffset(numberOfTreesInForest);

    for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
    {
        const Tree& tree = mForest.mTrees[treeId];
        BufferCollection& bc = perTreeBufferCollection[treeId];

        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFeature.mFloatParamsBufferId, tree.mFloatFeatureParams);
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mFeature.mIntParamsBufferId, tree.mIntFeatureParams);
        mPreSteps->ProcessStep(stack, bc, gen, bc, 0);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();

        if(useOobIndices)
        {
            oobIndices[treeId] = tree.mExtraInfo.GetBufferPtr< VectorBufferTemplate<typename BufferTypes::Index> >(OOB_INDICES);
            ASSERT(oobIndices[treeId]->IsSorted()) //Assuming OOB_INDICES have already been sorted
            currentOobOffset[treeId] = 0;
        }
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    ysOut.Resize(numberOfIndices, mCombiner.GetResultDim());

    for(typename BufferTypes::Index i=0; i<numberOfIndices; i++)
    {
        mCombiner.Reset();
        for(typename BufferTypes::Index treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            // Only include an datapoint if it is OOB or if we're not using OOB samples 
            const bool isOobIndex = useOobIndices && (oobIndices[treeId]->Get(currentOobOffset[treeId]) == i );
            ASSERT(!useOobIndices || (oobIndices[treeId]->Get(currentOobOffset[treeId]) < numberOfIndices))
            if(isOobIndex || !useOobIndices)
            {
                const Tree& tree = mForest.mTrees[treeId];
                typename BufferTypes::Index leafNodeId = walkTree<typename Feature::FeatureBinding, BufferTypes>(
                                                                    featureBindings[treeId], tree, 0, i);
                mCombiner.Combine(leafNodeId, tree.mCounts.Get(leafNodeId), tree.mYs);
            }

            if(isOobIndex)
            {
                currentOobOffset[treeId] = std::min(oobIndices[treeId]->GetN()-1, currentOobOffset[treeId]+1);
            }

        }
        mCombiner.WriteResult(i, ysOut);
    }
}

template <class Feature, class Combiner, class BufferTypes>
Forest TemplateForestPredictor<Feature, Combiner, BufferTypes>::GetForest() const
{
    return mForest;
}



