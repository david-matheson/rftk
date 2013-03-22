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
    return data_dict

def __setstate__(self,data_dict):
    self.__init__(data_dict['mPath'], data_dict['mIntFeatureParams'], data_dict['mFloatFeatureParams'], data_dict['mDepths'], data_dict['mCounts'], data_dict['mYs'])
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