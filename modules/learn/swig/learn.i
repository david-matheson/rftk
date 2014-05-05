%module learn
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferTypes.h"
    #include "TreeLearnerI.h"
    #include "DepthFirstTreeLearner.h"
    #include "BreadthFirstTreeLearner.h"
    #include "Biau2008TreeLearner.h"
    #include "ParallelForestLearner.h"
    #include "OnlineForestLearner.h"

    #include "LinearMatrixFeature.h"
    #include "ScaledDepthDeltaFeature.h"
    #include "ClassEstimatorUpdater.h"
    #include "ClassProbabilityOfError.h"
    #include "MeanVarianceEstimatorUpdater.h"
    #include "SumOfVarianceExpectedError.h"
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"

%import(module="rftk.matrix_features") "matrix_features_external.i"
%import(module="rftk.image_features") "image_features_external.i"
%import(module="rftk.classification") "classification_external.i"
%import(module="rftk.splitpoints") "splitpoints_external.i"

%include "TreeLearnerI.h"
%include "DepthFirstTreeLearner.h"
%include "BreadthFirstTreeLearner.h"
%include "Biau2008TreeLearner.h"
%include "ParallelForestLearner.h"
%include "OnlineForestLearner.h"

%include "LinearMatrixFeature.h"
%include "ScaledDepthDeltaFeature.h"
%include "ClassEstimatorUpdater.h"
%include "ClassProbabilityOfError.h"
%include "MeanVarianceEstimatorUpdater.h"
%include "SumOfVarianceExpectedError.h"

%template(DepthFirstTreeLearner_f32i32) DepthFirstTreeLearner<DefaultBufferTypes>;

%template(BreadthFirstTreeLearner_f32i32) BreadthFirstTreeLearner<DefaultBufferTypes>;

%template(Biau2008TreeLearner_f32i32) Biau2008TreeLearner<DefaultBufferTypes>;

%template(OnlineForestMatrixClassLearner_f32i32)  OnlineForestLearner< LinearMatrixFeature< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >, ClassEstimatorUpdater< DefaultBufferTypes >, ClassProbabilityOfError<DefaultBufferTypes>, DefaultBufferTypes >;

%template(OnlineForestMatrixClassLearner_Sparse_f32i32)  OnlineForestLearner< LinearMatrixFeature< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >, ClassEstimatorUpdater< DefaultBufferTypes >, ClassProbabilityOfError<DefaultBufferTypes>, DefaultBufferTypes >;

%template(OnlineForestScaledDepthDeltaClassLearner_f32i32)  OnlineForestLearner< ScaledDepthDeltaFeature< DefaultBufferTypes >, ClassEstimatorUpdater< DefaultBufferTypes >, ClassProbabilityOfError<DefaultBufferTypes>, DefaultBufferTypes >;

%template(OnlineForestMatrixRegressionLearner_f32i32)  OnlineForestLearner< LinearMatrixFeature< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >, MeanVarianceEstimatorUpdater< DefaultBufferTypes >, SumOfVarianceExpectedError<DefaultBufferTypes>, DefaultBufferTypes >;

%template(OnlineForestMatrixRegressionLearner_Sparse_f32i32)  OnlineForestLearner< LinearMatrixFeature< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >, MeanVarianceEstimatorUpdater< DefaultBufferTypes >, SumOfVarianceExpectedError<DefaultBufferTypes>, DefaultBufferTypes >;

%template(OnlineForestScaledDepthDeltaRegressionLearner_f32i32)  OnlineForestLearner< ScaledDepthDeltaFeature< DefaultBufferTypes >, MeanVarianceEstimatorUpdater< DefaultBufferTypes >, SumOfVarianceExpectedError<DefaultBufferTypes>, DefaultBufferTypes >;
