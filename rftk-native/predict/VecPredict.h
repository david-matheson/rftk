#pragma once

#include "MatrixBuffer.h"

#include "Forest.h"


class VecForestPredictor
{
public:
    VecForestPredictor( const Forest& forest );

    void PredictLeafs(const MatrixBufferFloat& x, MatrixBufferInt& leafsOut);
    void PredictYs(const MatrixBufferFloat& x, MatrixBufferFloat& ysOut);

    Forest mForest;
};

void VecPredictLeafs(const Forest& forest, const MatrixBufferFloat& x, MatrixBufferInt& leafsOut);
void VecPredictYs(const Forest& forest, const MatrixBufferFloat& x, MatrixBufferFloat& ysOut);