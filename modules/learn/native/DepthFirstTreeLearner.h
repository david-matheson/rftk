#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <limits>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"
#include "Tree.h"
#include "TrySplitCriteriaI.h"
#include "PipelineStepI.h"
#include "SplitSelectorI.h"
#include "TreeLearnerI.h"


template <class FloatType, class IntType>
class DepthFirstTreeLearner: public TreeLearnerI
{
public:
    DepthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                            const PipelineStepI* treeSteps,
                            const PipelineStepI* nodeSteps,
                            const SplitSelectorI<FloatType, IntType>* splitSelector);
    DepthFirstTreeLearner(const DepthFirstTreeLearner<FloatType, IntType> & other);

    virtual ~DepthFirstTreeLearner();

    virtual TreeLearnerI* Clone() const;
    virtual void Learn( const BufferCollection& data, Tree& tree, unsigned int seed ) const;

private:
    void ProcessNode( boost::mt19937& gen,
                      Tree& tree,
                      BufferCollectionStack& stack,
                      IntType nodeIndex,
                      IntType depth,
                      FloatType nodeSize ) const;

    const TrySplitCriteriaI* mTrySplitCriteria;
    const PipelineStepI* mTreeSteps;
    const PipelineStepI* mNodeSteps;
    const SplitSelectorI<FloatType, IntType>* mSplitSelector;
};

template <class FloatType, class IntType>
DepthFirstTreeLearner<FloatType, IntType>::DepthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                                                                  const PipelineStepI* treeSteps,
                                                                  const PipelineStepI* nodeSteps,
                                                                  const SplitSelectorI<FloatType, IntType>* splitSelector)
: mTrySplitCriteria( trySplitCriteria->Clone() )
, mTreeSteps( treeSteps->Clone() )
, mNodeSteps( nodeSteps->Clone() )
, mSplitSelector( splitSelector->Clone() )
{}

template <class FloatType, class IntType>
DepthFirstTreeLearner<FloatType, IntType>::DepthFirstTreeLearner(const DepthFirstTreeLearner<FloatType, IntType> & other)
: mTrySplitCriteria( other.mTrySplitCriteria->Clone() )
, mTreeSteps( other.mTreeSteps->Clone() )
, mNodeSteps( other.mNodeSteps->Clone() )
, mSplitSelector( other.mSplitSelector->Clone() )
{
}

template <class FloatType, class IntType>
DepthFirstTreeLearner<FloatType, IntType>::~DepthFirstTreeLearner()
{
    delete mTrySplitCriteria;
    delete mTreeSteps;
    delete mNodeSteps;
    delete mSplitSelector;
}

template <class FloatType, class IntType>
TreeLearnerI* DepthFirstTreeLearner<FloatType, IntType>::Clone() const
{
    TreeLearnerI* clone = new DepthFirstTreeLearner(*this);
    return clone;
}


template <class FloatType, class IntType>
void DepthFirstTreeLearner<FloatType, IntType>::Learn( const BufferCollection& data, Tree& tree, unsigned int seed ) const
{
    boost::mt19937 gen;
    gen.seed(seed);

    BufferCollectionStack stack;
    stack.Push(&data);
    BufferCollection treeData;
    stack.Push(&treeData);
    mTreeSteps->ProcessStep(stack, treeData, gen);

    //emptyIndicesCollection is pushed so subsequent leftIndicesBufCol and rightIndicesBufCol
    //can be popped before adding the next layer down
    BufferCollection emptyIndicesCollection;
    stack.Push(&emptyIndicesCollection);
    ProcessNode(gen, tree, stack, 0, 0, std::numeric_limits<FloatType>::max());
}

template <class FloatType, class IntType>
void DepthFirstTreeLearner<FloatType, IntType>::ProcessNode( boost::mt19937& gen,
                                                              Tree& tree, BufferCollectionStack& stack,
                                                              IntType nodeIndex,
                                                              IntType depth,
                                                              FloatType nodeSize ) const
{
    if(mTrySplitCriteria->TrySplit(depth, nodeSize))
    {
        bool doSplit = false;
        BufferCollection leftIndicesBufCol;
        BufferCollection rightIndicesBufCol;
        FloatType leftSize = std::numeric_limits<FloatType>::min();
        FloatType rightSize = std::numeric_limits<FloatType>::min();
        IntType leftNodeIndex = -1;
        IntType rightNodeIndex = -1;

        // Using a nested block so memory is freed before recursing
        {
            BufferCollection nodeData;
            stack.Push(&nodeData);
            mNodeSteps->ProcessStep(stack, nodeData, gen);
            SplitSelectorInfo<FloatType, IntType> selectorInfo = mSplitSelector->ProcessSplits(stack, depth);
            doSplit = selectorInfo.ValidSplit();
            if(doSplit)
            {
                leftNodeIndex = tree.NextNodeIndex();
                rightNodeIndex = tree.NextNodeIndex();

                selectorInfo.WriteToTree( nodeIndex, leftNodeIndex, rightNodeIndex,
                                          tree.mCounts, tree.mDepths, tree.mFloatFeatureParams, tree.mIntFeatureParams, tree.mYs);

                tree.mPath.Set(nodeIndex, 0, leftNodeIndex);
                tree.mPath.Set(nodeIndex, 1, rightNodeIndex);

                selectorInfo.SplitIndices(leftIndicesBufCol, rightIndicesBufCol, leftSize, rightSize);
            }
            stack.Pop(); //stack.Push(nodeData);
        }

        if( doSplit )
        {
            stack.Pop(); //stack.Push(&emptyIndicesCollection); or stack.Push(&leftIndicesBufCol); or stack.Push(&rightIndicesBufCol);

            stack.Push(&leftIndicesBufCol);
            ProcessNode(gen, tree, stack, leftNodeIndex, depth+1, leftSize);

            stack.Pop(); //stack.Push(&emptyIndicesCollection); or stack.Push(&leftIndicesBufCol); or stack.Push(&rightIndicesBufCol);

            stack.Push(&rightIndicesBufCol);
            ProcessNode(gen, tree, stack, rightNodeIndex, depth+1, rightSize);
        }
    }
}