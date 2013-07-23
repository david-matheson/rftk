#include "BufferTypes.h"
#include "DepthFirstTreeLearner.h"
#include "BreadthFirstTreeLearner.h"
#include "OnlineForestLearner.h"
#include "ProbabilityOfErrorFrontierQueue.h"
#include "Biau2008TreeLearner.h"

#include "LinearMatrixFeature.h"
#include "ClassEstimatorUpdater.h"
#include "ClassProbabilityOfError.h"

template class OnlineForestLearner< LinearMatrixFeature< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >, ClassEstimatorUpdater< float, int >, ClassProbabilityOfError, float, int >;