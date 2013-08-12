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
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
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

%template(SplitSelectorI_f32i32) SplitSelectorI<float, int>;
%template(SplitSelector_f32i32) SplitSelector<float, int>;
%template(WaitForBestSplitSelector_f32i32) WaitForBestSplitSelector<float, int>;
%template(RandomSplitpointsStep_f32i32) RandomSplitpointsStep<DefaultBufferTypes>;

%template(AssignStreamStep_f32i32) AssignStreamStep<DefaultBufferTypes>;
%template(RangeMidpointStep_f32i32) RangeMidpointStep<DefaultBufferTypes>;

%template(SplitIndices_f32i32) SplitBuffersIndices<float, int>;
%template(SplitBuffersFeatureRange_f32i32) SplitBuffersFeatureRange<DefaultBufferTypes>;