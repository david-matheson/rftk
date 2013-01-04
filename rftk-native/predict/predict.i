%module predict
%{
    #define SWIG_FILE_WITH_INIT
    #include "Forest.h"
    #include "VecPredict.h"
%}

%include "std_vector.i"

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"

namespace std {
    %template(TreesVector) std::vector<Tree>;
}

%include "Forest.h"
%include "VecPredict.h"
