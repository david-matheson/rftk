%module train
%{
    #define SWIG_FILE_WITH_INIT
    #include "SplitCriteriaI.h"
    #include "NodeDataCollectorI.h"
    #include "TrainConfigParams.h"
    #include "OfflineSamplingParams.h"
    #include "OnlineSamplingParams.h"
    #include "ActiveSplitNode.h"
    #include "DepthFirstParallelForestLearner.h"
    #include "OnlineForestLearner.h"
    #include "AllNodeDataCollector.h"
    #include "RandomThresholdHistogramDataCollector.h"
    #include "TwoStreamRandomThresholdHistogramDataCollector.h"
    #include "OfflineSplitCriteria.h"
    #include "OnlineAlphaBetaSplitCriteria.h"
    #include "OnlineConsistentSplitCriteria.h"
%}

%include <exception.i>
%import "asserts/asserts.i"
%import "buffers/buffers.i"


%include "std_vector.i"

namespace std {
    %template(FeatureExtractorVector) std::vector<FeatureExtractorI*>;
}

%include "SplitCriteriaI.h"
%include "NodeDataCollectorI.h"
%include "TrainConfigParams.h"
%include "OfflineSamplingParams.h"
%include "OnlineSamplingParams.h"
%include "ActiveSplitNode.h"
%include "DepthFirstParallelForestLearner.h"
%include "OnlineForestLearner.h"
%include "AllNodeDataCollector.h"
%include "RandomThresholdHistogramDataCollector.h"
%include "TwoStreamRandomThresholdHistogramDataCollector.h"
%include "OfflineSplitCriteria.h"
%include "OnlineAlphaBetaSplitCriteria.h"
%include "OnlineConsistentSplitCriteria.h"