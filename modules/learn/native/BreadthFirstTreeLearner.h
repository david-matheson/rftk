#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <limits>
#include <list>
#include <queue>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"
#include "Tree.h"
#include "TrySplitCriteriaI.h"
#include "PipelineStepI.h"
#include "SplitSelectorI.h"
#include "TreeLearnerI.h"
#include "ActiveLeaf.h"

template <class FloatType, class IntType>
class BreadthFirstTreeLearner: public TreeLearnerI
{
public:
    BreadthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                            const PipelineStepI* treeSteps,
                            const PipelineStepI* nodeSteps,
                            const SplitSelectorI<FloatType, IntType>* splitSelector);
    BreadthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                            const PipelineStepI* treeSteps,
                            const PipelineStepI* nodeSteps,
                            const SplitSelectorI<FloatType, IntType>* splitSelector,
                            const IntType maxNumberOfLeaves);
    BreadthFirstTreeLearner(const BreadthFirstTreeLearner<FloatType, IntType> & other);

    virtual ~BreadthFirstTreeLearner();

    virtual TreeLearnerI* Clone() const;
    virtual void Learn( BufferCollectionStack stack, Tree& tree, unsigned int seed ) const;

private:
    bool ProcessActiveLeaf( boost::mt19937& gen,
                              Tree& tree,
                              BufferCollectionStack& stack,
                              std::queue< ActiveLeaf<FloatType, IntType> >& activeLeaves ) const;

    const TrySplitCriteriaI* mTrySplitCriteria;
    const PipelineStepI* mTreeSteps;
    const PipelineStepI* mNodeSteps;
    const SplitSelectorI<FloatType, IntType>* mSplitSelector;
    const IntType mMaxNumberOfSplits;
};

template <class FloatType, class IntType>
BreadthFirstTreeLearner<FloatType, IntType>::BreadthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                                                                const PipelineStepI* treeSteps,
                                                                const PipelineStepI* nodeSteps,
                                                                const SplitSelectorI<FloatType, IntType>* splitSelector)
: mTrySplitCriteria( trySplitCriteria->Clone() )
, mTreeSteps( treeSteps->Clone() )
, mNodeSteps( nodeSteps->Clone() )
, mSplitSelector( splitSelector->Clone() )
, mMaxNumberOfSplits( std::numeric_limits<IntType>::max() )
{}

template <class FloatType, class IntType>
BreadthFirstTreeLearner<FloatType, IntType>::BreadthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                                                                const PipelineStepI* treeSteps,
                                                                const PipelineStepI* nodeSteps,
                                                                const SplitSelectorI<FloatType, IntType>* splitSelector,
                                                                const IntType maxNumberOfLeaves)
: mTrySplitCriteria( trySplitCriteria->Clone() )
, mTreeSteps( treeSteps->Clone() )
, mNodeSteps( nodeSteps->Clone() )
, mSplitSelector( splitSelector->Clone() )
, mMaxNumberOfSplits( maxNumberOfLeaves )
{}

template <class FloatType, class IntType>
BreadthFirstTreeLearner<FloatType, IntType>::BreadthFirstTreeLearner(const BreadthFirstTreeLearner<FloatType, IntType> & other)
: mTrySplitCriteria( other.mTrySplitCriteria->Clone() )
, mTreeSteps( other.mTreeSteps->Clone() )
, mNodeSteps( other.mNodeSteps->Clone() )
, mSplitSelector( other.mSplitSelector->Clone() )
, mMaxNumberOfSplits( other.mMaxNumberOfSplits )
{
}

template <class FloatType, class IntType>
BreadthFirstTreeLearner<FloatType, IntType>::~BreadthFirstTreeLearner()
{
    delete mTrySplitCriteria;
    delete mTreeSteps;
    delete mNodeSteps;
    delete mSplitSelector;
}

template <class FloatType, class IntType>
TreeLearnerI* BreadthFirstTreeLearner<FloatType, IntType>::Clone() const
{
    TreeLearnerI* clone = new BreadthFirstTreeLearner(*this);
    return clone;
}


template <class FloatType, class IntType>
void BreadthFirstTreeLearner<FloatType, IntType>::Learn( BufferCollectionStack stack, Tree& tree, unsigned int seed ) const
{
    boost::mt19937 gen;
    gen.seed(seed);

    BufferCollection treeData;
    stack.Push(&treeData);
    mTreeSteps->ProcessStep(stack, treeData, gen);

    std::queue< ActiveLeaf<FloatType, IntType> > activeLeaves;
    activeLeaves.push(ActiveLeaf<FloatType, IntType>(0, 0));

    // FIFO breath first
    int numberOfSplits = 0;
    while(numberOfSplits < mMaxNumberOfSplits && activeLeaves.size() > 0)
    {
        const bool didSplit = ProcessActiveLeaf(gen, tree, stack, activeLeaves);
        if( didSplit )
        {
            numberOfSplits++; 
        }
    }
}

template <class FloatType, class IntType>
bool BreadthFirstTreeLearner<FloatType, IntType>::ProcessActiveLeaf( boost::mt19937& gen,
                                                              Tree& tree,
                                                              BufferCollectionStack& stack,
                                                              std::queue< ActiveLeaf<FloatType, IntType> >& activeLeaves) const

{
    ActiveLeaf<FloatType, IntType>& activeLeaf = activeLeaves.front();

    stack.Push(&activeLeaf.mSplitBufferCollection);
    BufferCollection nodeData;
    stack.Push(&nodeData);
    mNodeSteps->ProcessStep(stack, nodeData, gen);
    SplitSelectorInfo<FloatType, IntType> selectorInfo = mSplitSelector->ProcessSplits(stack, activeLeaf.mDepth);

    const bool ValidSplit = selectorInfo.ValidSplit(); 
    if(ValidSplit) 
    {
        const IntType leftNodeIndex = tree.NextNodeIndex();
        const IntType rightNodeIndex = tree.NextNodeIndex();

        selectorInfo.WriteToTree( activeLeaf.mNodeIndex, leftNodeIndex, rightNodeIndex,
                                  tree.mCounts, tree.mDepths, tree.mFloatFeatureParams, tree.mIntFeatureParams, tree.mYs);

        tree.mPath.Set(activeLeaf.mNodeIndex, 0, leftNodeIndex);
        tree.mPath.Set(activeLeaf.mNodeIndex, 1, rightNodeIndex);

        ActiveLeaf<FloatType, IntType> leftActiveLeaf(leftNodeIndex, activeLeaf.mDepth+1);
        ActiveLeaf<FloatType, IntType> rightActiveLeaf(rightNodeIndex, activeLeaf.mDepth+1);
        FloatType leftSize = std::numeric_limits<FloatType>::min();
        FloatType rightSize = std::numeric_limits<FloatType>::min();
        selectorInfo.SplitBuffers(leftActiveLeaf.mSplitBufferCollection, rightActiveLeaf.mSplitBufferCollection, leftSize, rightSize);

        if(mTrySplitCriteria->TrySplit(leftActiveLeaf.mDepth, leftSize))
        {
            activeLeaves.push(leftActiveLeaf);
        }
        if(mTrySplitCriteria->TrySplit(rightActiveLeaf.mDepth, rightSize))
        {
            activeLeaves.push(rightActiveLeaf);
        }
    }

    // Remove the node
    activeLeaves.pop();

    stack.Pop(); //stack.Push(activeLeaf.mSplitBufferCollection)
    stack.Pop(); //stack.Push(&nodeData);

    return ValidSplit;
}