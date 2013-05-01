%module should_split
%{
    #define SWIG_FILE_WITH_INIT
    #include "ShouldSplitCriteriaI.h"
    #include "MinChildSizeCriteria.h"
    #include "MinImpurityCriteria.h"
    #include "ShouldSplitCombinedCriteria.h"
    #include "ShouldSplitNoCriteria.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"

%include "ShouldSplitCriteriaI.h"
%include "MinChildSizeCriteria.h"
%include "MinImpurityCriteria.h"
%include "ShouldSplitCombinedCriteria.h"
%include "ShouldSplitNoCriteria.h"