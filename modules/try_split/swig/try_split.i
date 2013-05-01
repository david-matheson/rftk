%module try_split
%{
    #define SWIG_FILE_WITH_INIT
    #include "TrySplitCriteriaI.h"
    #include "MaxDepthCriteria.h"
    #include "MinNodeSizeCriteria.h"
    #include "TimeLimitCriteria.h"
    #include "TrySplitCombinedCriteria.h"
    #include "TrySplitNoCriteria.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"

%include "std_vector.i"

namespace std {
    %template(TrySplitCriteriaVector) std::vector<TrySplitCriteriaI*>;
}


%include "TrySplitCriteriaI.h"
%include "MaxDepthCriteria.h"
%include "MinNodeSizeCriteria.h"
%include "TimeLimitCriteria.h"
%include "TrySplitCombinedCriteria.h"
%include "TrySplitNoCriteria.h"