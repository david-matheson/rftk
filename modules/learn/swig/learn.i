%module learn
%{
    #define SWIG_FILE_WITH_INIT
    #include "TreeLearnerI.h"
    #include "DepthFirstTreeLearner.h"
    #include "ParallelForestLearner.h"
    #include "OnlineForestLearner.h"

    #include "LinearMatrixFeature.h"
    #include "ScaledDepthDeltaFeature.h"
    #include "ClassEstimatorUpdater.h"
    #include "ClassProbabilityOfError.h"


%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"

%import(module="rftk.matrix_features") "matrix_features_external.i"
%import(module="rftk.image_features") "image_features_external.i"
%import(module="rftk.classification") "classification_external.i"
%import(module="rftk.splitpoints") "splitpoints_external.i"

%include "TreeLearnerI.h"
%include "DepthFirstTreeLearner.h"
%include "ParallelForestLearner.h"
%include "OnlineForestLearner.h"

%include "LinearMatrixFeature.h"
#include "ScaledDepthDeltaFeature.h"
%include "ClassEstimatorUpdater.h"
%include "ClassProbabilityOfError.h"

%template(DepthFirstTreeLearner_f32i32) DepthFirstTreeLearner<float, int>;

%template(OnlineForestMatrixClassLearner_f32i32)  OnlineForestLearner< LinearMatrixFeature< MatrixBufferTemplate<float>, float, int >, ClassEstimatorUpdater< float, int >, ClassProbabilityOfError, float, int >;

%template(OnlineForestScaledDepthDeltaClassLearner_f32i32)  OnlineForestLearner< ScaledDepthDeltaFeature< float, int >, ClassEstimatorUpdater< float, int >, ClassProbabilityOfError, float, int >;