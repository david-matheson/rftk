#include "assert_util.h"

#include "Tree.h"

Tree::Tree( const MatrixBufferInt& path,
            const MatrixBufferInt& intFeatureParams,
            const MatrixBufferFloat& floatFeatureParams,
            const MatrixBufferInt& depths,
            const MatrixBufferFloat& ys )
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
