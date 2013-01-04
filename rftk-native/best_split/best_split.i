%module best_split
%{
    #define SWIG_FILE_WITH_INIT
    #include "BestSplitI.h"
    #include "ClassInfoGainAllThresholdsBestSplit.h"
%}

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"

%include "BestSplitI.h"
%include "ClassInfoGainAllThresholdsBestSplit.h"

