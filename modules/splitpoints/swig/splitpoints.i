%module splitpoints
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferTypes.h"
    #include "RandomSplitpointsStep.h"
    #include "AssignStreamStep.h"
    #include "RangeMidpointStep.h"
    #include "SplitSelectorBuffers.h"

    #include "SplitBuffersI.h"
    #include "SplitBuffersIndices.h"
    #include "SplitBuffersFeatureRange.h"
    #include "SplitBuffersList.h"
    #include "BestSplitpointsWalkingSortedStep.h"
    #include "DownSampleBestSplitpointsWalkingSortedStep.h"
    #include "RandomUniformSplitpointsInRangeStep.h"

    #if PY_VERSION_HEX >= 0x03020000
    # define SWIGPY_SLICE_ARG(obj) ((PyObject*) (obj))
    #else
    # define SWIGPY_SLICE_ARG(obj) ((PySliceObject*) (obj))
    #endif
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%include <splitpoints_external.i>

%include "std_vector.i"

namespace std {
    %template(SplitBuffersVector) std::vector<SplitBuffersI*>;
    %template(SplitSelectorBufferVector) std::vector<SplitSelectorBuffers>;
}

%include "RandomSplitpointsStep.h"
%include "AssignStreamStep.h"
%include "RangeMidpointStep.h"
%include "SplitSelectorBuffers.h"

%include "SplitBuffersI.h"
%include "SplitBuffersIndices.h"
%include "SplitBuffersFeatureRange.h"
%include "SplitBuffersList.h"
%include "RandomUniformSplitpointsInRangeStep.h"
%include "BestSplitpointsWalkingSortedStep.h"
%include "DownSampleBestSplitpointsWalkingSortedStep.h"

%template(SplitSelectorI_f32i32) SplitSelectorI<DefaultBufferTypes>;
%template(SplitSelector_f32i32) SplitSelector<DefaultBufferTypes>;
%template(WaitForBestSplitSelector_f32i32) WaitForBestSplitSelector<DefaultBufferTypes>;
%template(RandomSplitpointsStep_f32i32) RandomSplitpointsStep<DefaultBufferTypes>;

%template(AssignStreamStep_f32i32) AssignStreamStep<DefaultBufferTypes>;
%template(RangeMidpointStep_f32i32) RangeMidpointStep<DefaultBufferTypes>;

%template(SplitIndices_f32i32) SplitBuffersIndices<DefaultBufferTypes>;
%template(SplitBuffersFeatureRange_f32i32) SplitBuffersFeatureRange<DefaultBufferTypes>;
%template(RandomUniformSplitpointsInRangeStep_Default) RandomUniformSplitpointsInRangeStep< DefaultBufferTypes >;