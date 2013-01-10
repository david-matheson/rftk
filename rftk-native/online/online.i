%module online
%{
    #define SWIG_FILE_WITH_INIT
    #include "OnlineLeafFactoryI.h"
    #include "OnlineLeafFactories.h"
    #include "OnlineLeafI.h"
    #include "AllThresholdsOnlineLeaf.h"
    #include "RandomThresholdsOnlineLeaf.h"
    #include "AlphaBetaSplitCriteria.h"
    #include "HoeffdingSplitCriteria.h"
    #include "OnlineLeafSet.h"
    #include "OnlineTree.h"
%}

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"

%include "OnlineLeafFactoryI.h"
%include "OnlineLeafFactories.h"
%include "OnlineLeafI.h"
%include "AllThresholdsOnlineLeaf.h"
%include "RandomThresholdsOnlineLeaf.h"
%include "AlphaBetaSplitCriteria.h"
%include "HoeffdingSplitCriteria.h"
%include "OnlineLeafSet.h"
%include "OnlineTree.h"
