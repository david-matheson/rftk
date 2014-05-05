#include <stdio.h>
#include <limits>

#include <asserts.h>
#include "Tree.h"
#include "ForestStats.h"

TreeData::TreeData()
: mPath(0,0, NULL_CHILD)
, mIntFeatureParams(0,0)
, mFloatFeatureParams(0,0)
, mCounts(0)
, mDepths(0)
, mYs(0,0)
, mExtraInfo()
, mLastNodeIndex(0)
, mValid(false)
{
}


TreeData::TreeData( const MatrixBufferTemplate<int>& path,
            const MatrixBufferTemplate<int>& intFeatureParams,
            const MatrixBufferTemplate<float>& floatFeatureParams,
            const VectorBufferTemplate<int> & depths,
            const VectorBufferTemplate<float>& counts,
            const MatrixBufferTemplate<float>& ys )
: mPath(path)
, mIntFeatureParams(intFeatureParams)
, mFloatFeatureParams(floatFeatureParams)
, mCounts(counts)
, mDepths(depths)
, mYs(ys)
, mExtraInfo()
, mLastNodeIndex(mPath.GetM())
, mValid(true)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mDepths.GetN())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mCounts.GetN())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())
}


TreeData::TreeData( int initalNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim  )
: mPath(initalNumberNodes, 2, NULL_CHILD)
, mIntFeatureParams(initalNumberNodes, maxIntParamsDim)
, mFloatFeatureParams(initalNumberNodes, maxFloatParamsDim)
, mCounts(initalNumberNodes)
, mDepths(initalNumberNodes)
, mYs(initalNumberNodes, maxYsDim)
, mExtraInfo()
, mLastNodeIndex(1)
, mValid(true)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())

    //TODO: This needs to be changed for regression
    mYs.SetAll(1.0f/static_cast<float>(mYs.GetN()));
}

TreeData::TreeData(const TreeData& tree)
: mPath(tree.mPath)
, mIntFeatureParams(tree.mIntFeatureParams)
, mFloatFeatureParams(tree.mFloatFeatureParams)
, mCounts(tree.mCounts)
, mDepths(tree.mDepths)
, mYs(tree.mYs)
, mExtraInfo(tree.mExtraInfo)
, mLastNodeIndex(tree.mLastNodeIndex)
, mValid(tree.mValid)
{
}

TreeData::~TreeData()
{
}

int TreeData::NextNodeIndex()
{
    int nextNodeIndex = mLastNodeIndex;
    mLastNodeIndex++;
    const int numberOfNodesAllocated = mPath.GetM();
    if( mLastNodeIndex >= numberOfNodesAllocated )
    {
        int newNumberOfNodesAllocated = mLastNodeIndex + numberOfNodesAllocated/2 + 1;
        mPath.Resize(newNumberOfNodesAllocated, 2, NULL_CHILD);
        mIntFeatureParams.Resize(newNumberOfNodesAllocated, mIntFeatureParams.GetN());
        mFloatFeatureParams.Resize(newNumberOfNodesAllocated, mFloatFeatureParams.GetN());
        mCounts.Resize(newNumberOfNodesAllocated);
        mDepths.Resize(newNumberOfNodesAllocated);
        mYs.Resize(newNumberOfNodesAllocated, mYs.GetN());
    }
    return nextNodeIndex;
}

void TreeData::Compact()
{
    mPath.Resize(mLastNodeIndex, 2, NULL_CHILD);
    mIntFeatureParams.Resize(mLastNodeIndex, mIntFeatureParams.GetN());
    mFloatFeatureParams.Resize(mLastNodeIndex, mFloatFeatureParams.GetN());
    mCounts.Resize(mLastNodeIndex);
    mDepths.Resize(mLastNodeIndex);
    mYs.Resize(mLastNodeIndex, mYs.GetN());
}

Tree::Tree()
: mTreeData(new TreeData())
{
}


Tree::Tree( const MatrixBufferTemplate<int>& path,
            const MatrixBufferTemplate<int>& intFeatureParams,
            const MatrixBufferTemplate<float>& floatFeatureParams,
            const VectorBufferTemplate<int> & depths,
            const VectorBufferTemplate<float>& counts,
            const MatrixBufferTemplate<float>& ys )
: mTreeData(new TreeData(path, intFeatureParams, floatFeatureParams, depths, counts, ys))
{
}


Tree::Tree( int initalNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim  )
: mTreeData(new TreeData(initalNumberNodes, maxIntParamsDim, maxFloatParamsDim, maxYsDim))
{
}

Tree::~Tree()
{
}


void Tree::GatherStats(ForestStats& stats) const
{
    for(int nodeId=0; nodeId<mTreeData->mLastNodeIndex-1; nodeId++)
    {
        const bool isLeaf = (mTreeData->mPath.Get(nodeId, 0) == NULL_CHILD
                             || mTreeData->mPath.Get(nodeId, 1) == NULL_CHILD);
        if( isLeaf )
        {
            // printf("TreeData::GatherStats %d %d %d\n", nodeId, depth, numberEstimatorSamples);
            stats.ProcessLeaf(*this, nodeId);
        }
    }
}

int Tree::NextNodeIndex()
{
    return mTreeData->NextNodeIndex();
}

void Tree::Compact()
{
    mTreeData->Compact();
}

MatrixBufferTemplate<int>& Tree::GetPath()
{
    return mTreeData->mPath;
}

MatrixBufferTemplate<int>& Tree::GetIntFeatureParams()
{
   return mTreeData->mIntFeatureParams;
}

MatrixBufferTemplate<float>& Tree::GetFloatFeatureParams()
{
   return mTreeData->mFloatFeatureParams;
}

VectorBufferTemplate<float>& Tree::GetCounts()
{
   return mTreeData->mCounts;
}

VectorBufferTemplate<int>& Tree::GetDepths()
{
   return mTreeData->mDepths;
}

MatrixBufferTemplate<float>& Tree::GetYs()
{
   return mTreeData->mYs;
}

BufferCollection& Tree::GetExtraInfo()
{
   return mTreeData->mExtraInfo;
}

const MatrixBufferTemplate<int>& Tree::GetPath() const
{
    return mTreeData->mPath;
}

const MatrixBufferTemplate<int>& Tree::GetIntFeatureParams() const
{
   return mTreeData->mIntFeatureParams;
}

const MatrixBufferTemplate<float>& Tree::GetFloatFeatureParams() const
{
   return mTreeData->mFloatFeatureParams;
}

const VectorBufferTemplate<float>& Tree::GetCounts() const
{
   return mTreeData->mCounts;
}

const VectorBufferTemplate<int>& Tree::GetDepths() const
{
   return mTreeData->mDepths;
}

const MatrixBufferTemplate<float>& Tree::GetYs() const
{
   return mTreeData->mYs;
}

const BufferCollection& Tree::GetExtraInfo() const
{
   return mTreeData->mExtraInfo;
}