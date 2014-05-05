%module try_split
%{
    #define SWIG_FILE_WITH_INIT
    #include "TrySplitCriteriaI.h"
    #include "MaxDepthCriteria.h"
    #include "MinNodeSizeCriteria.h"
    #include "TimeLimitCriteria.h"
    #include "TrySplitCombinedCriteria.h"
    #include "TrySplitNoCriteria.h"

    #if PY_VERSION_HEX >= 0x03020000
    # define SWIGPY_SLICE_ARG(obj) ((PyObject*) (obj))
    #else
    # define SWIGPY_SLICE_ARG(obj) ((PySliceObject*) (obj))
    #endif
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
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