%module forest_data
%{
    #define SWIG_FILE_WITH_INIT
    #include "Tree.h"
    #include "ForestStats.h"
    #include "Forest.h"

    #if PY_VERSION_HEX >= 0x03020000
    # define SWIGPY_SLICE_ARG(obj) ((PyObject*) (obj))
    #else
    # define SWIGPY_SLICE_ARG(obj) ((PySliceObject*) (obj))
    #endif
%}

%include <exception.i>
%include "numpy.i"
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"

%include "std_vector.i"

%init %{
    import_array();
%}

namespace std {
    %template(TreesVector) std::vector<Tree>;
}

%include "Tree.h"
%include "ForestStats.h"
%include "Forest.h"

%extend Tree {
%insert("python") %{

def __getstate__(self):
    self.Compact()
    data_dict = {}
    data_dict['mPath'] = self.mPath
    data_dict['mIntFeatureParams'] = self.mIntFeatureParams
    data_dict['mFloatFeatureParams'] = self.mFloatFeatureParams
    data_dict['mDepths'] = self.mDepths
    data_dict['mCounts'] = self.mCounts
    data_dict['mYs'] = self.mYs

    data_dict['mExtraInfo'] = {}
    keys = self.mExtraInfo.GetKeys()
    for key in keys:
        data_dict['mExtraInfo'][key] = self.mExtraInfo.GetBuffer(key) 

    return data_dict

def __setstate__(self,data_dict):
    self.__init__(data_dict['mPath'], data_dict['mIntFeatureParams'], data_dict['mFloatFeatureParams'], data_dict['mDepths'], data_dict['mCounts'], data_dict['mYs'])
    for key, value in data_dict['mExtraInfo'].iteritems():
        self.mExtraInfo.AddBuffer(key, value)

%}
}

%extend Forest {
%insert("python") %{

def __getstate__(self):
    trees = []
    for tree_id in range(self.GetNumberOfTrees()):
        trees.append(self.GetTree(tree_id))
    data_dict = {}
    data_dict['trees'] = trees
    return data_dict

def __setstate__(self,data_dict):
    self.__init__(data_dict['trees'])
%}
}