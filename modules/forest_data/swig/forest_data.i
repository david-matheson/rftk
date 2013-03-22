%module forest_data
%{
    #define SWIG_FILE_WITH_INIT
    #include "Tree.h"
    #include "Forest.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"

%include "std_vector.i"

namespace std {
    %template(TreesVector) std::vector<Tree>;
}

%include "Tree.h"
%include "Forest.h"