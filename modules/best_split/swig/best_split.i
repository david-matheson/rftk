%module best_split
%{
    #define SWIG_FILE_WITH_INIT
    #include "BestSplitI.h"
    #include "ClassInfoGainAllThresholdsBestSplit.h"
    #include "ClassInfoGainHistogramsBestSplit.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"

%include "BestSplitI.h"
%include "ClassInfoGainAllThresholdsBestSplit.h"
%include "ClassInfoGainHistogramsBestSplit.h"
