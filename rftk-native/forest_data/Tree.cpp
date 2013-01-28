#include "assert_util.h"

#include "Tree.h"

Tree::Tree( const Int32MatrixBuffer& path,
            const Int32MatrixBuffer& intFeatureParams,
            const Float32MatrixBuffer& floatFeatureParams,
            const Int32MatrixBuffer& depths,
            const Float32MatrixBuffer& ys )
: mPath(path)
, mIntFeatureParams(intFeatureParams)
, mFloatFeatureParams(floatFeatureParams)
, mDepths(depths)
, mYs(ys)
, mValid(true)
, mLastNodeIndex(0)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mDepths.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())
}

Tree::Tree( int maxNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim  )
: mPath(maxNumberNodes, 2)
, mIntFeatureParams(maxNumberNodes, maxIntParamsDim)
, mFloatFeatureParams(maxNumberNodes, maxFloatParamsDim)
, mDepths(maxNumberNodes, 1)
, mYs(maxNumberNodes, maxYsDim)
, mValid(true)
, mLastNodeIndex(0)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())

    mPath.SetAll(-1);
}
