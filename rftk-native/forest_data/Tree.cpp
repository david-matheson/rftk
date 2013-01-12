#include "assert_util.h"

#include "Tree.h"

Tree::Tree( const MatrixBufferInt& path,
            const MatrixBufferInt& intFeatureParams,
            const MatrixBufferFloat& floatFeatureParams,
            const MatrixBufferFloat& ys )
: mPath(path)
, mIntFeatureParams(intFeatureParams)
, mFloatFeatureParams(floatFeatureParams)
, mYs(ys)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())
}

