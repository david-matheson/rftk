%module train
%{
    #define SWIG_FILE_WITH_INIT
    #include "SplitCriteriaI.h"
    #include "NodeDataCollectorI.h"
    #include "TrainConfigParams.h"
    #include "OfflineSamplingParams.h"
    #include "ActiveSplitNode.h"
    #include "DepthFirstParallelForestLearner.h"
    #include "OnlineForestLearner.h"
%}

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"

%include "SplitCriteriaI.h"
%include "NodeDataCollectorI.h"
%include "TrainConfigParams.h"
%include "OfflineSamplingParams.h"
%include "ActiveSplitNode.h"
%include "DepthFirstParallelForestLearner.h"
%include "OnlineForestLearner.h"