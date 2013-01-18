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
    #include "OfflineSplitCriteria.h"
    #include "OnlineAlphaBetaSplitCriteria.h"
%}

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"


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
%include "OfflineSplitCriteria.h"
%include "OnlineAlphaBetaSplitCriteria.h"