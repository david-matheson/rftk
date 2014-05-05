%module should_split
%{
    #define SWIG_FILE_WITH_INIT
    #include "ShouldSplitCriteriaI.h"
    #include "MinChildSizeCriteria.h"
    #include "MinChildSizeSumCriteria.h"
    #include "MinImpurityCriteria.h"
    #include "OnlineConsistentCriteria.h"
    #include "ShouldSplitCombinedCriteria.h"
    #include "ShouldSplitNoCriteria.h"

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
    %template(ShouldSplitCriteriaVector) std::vector<ShouldSplitCriteriaI*>;
}


%include "ShouldSplitCriteriaI.h"
%include "MinChildSizeCriteria.h"
%include "MinChildSizeSumCriteria.h"
%include "MinImpurityCriteria.h"
%include "OnlineConsistentCriteria.h"
%include "ShouldSplitCombinedCriteria.h"
%include "ShouldSplitNoCriteria.h"