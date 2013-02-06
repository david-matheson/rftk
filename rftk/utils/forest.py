import cPickle as pickle
import numpy as np

import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.forest_data as forest_data
import rftk.utils.buffer_converters as buffer_converters

class PyTree(object):
    pass

def as_pyforest(native_forest):
    pytrees = []
    for tree_id in range(native_forest.GetNumberOfTrees()):
        native_tree = native_forest.GetTree(tree_id)
        pytree = PyTree()
        pytree.path = buffer_converters.as_numpy_array(native_tree.mPath)[0:native_tree.mLastNodeIndex+1,:]
        pytree.int_features_params = buffer_converters.as_numpy_array(native_tree.mIntFeatureParams)[0:native_tree.mLastNodeIndex+1,:]
        pytree.float_features_params = buffer_converters.as_numpy_array(native_tree.mFloatFeatureParams)[0:native_tree.mLastNodeIndex+1,:]       
        pytree.depths = buffer_converters.as_numpy_array(native_tree.mDepths, flatten=True)[0:native_tree.mLastNodeIndex+1]
        pytree.counts = buffer_converters.as_numpy_array(native_tree.mCounts, flatten=True)[0:native_tree.mLastNodeIndex+1] 
        pytree.ys = buffer_converters.as_numpy_array(native_tree.mYs)[0:native_tree.mLastNodeIndex+1,:]
        pytrees.append(pytree)
    return pytrees

def as_nativeforest(pytrees):
    native_trees = []
    for tree in pytrees:
        native_tree = forest_data.Tree(buffer_converters.as_matrix_buffer(tree.path),
                                        buffer_converters.as_matrix_buffer(tree.int_features_params),
                                        buffer_converters.as_matrix_buffer(tree.float_features_params),
                                        buffer_converters.as_vector_buffer(tree.depths),
                                        buffer_converters.as_vector_buffer(tree.counts),
                                        buffer_converters.as_matrix_buffer(tree.ys))
        native_trees.append(native_tree)
    return forest_data.Forest(native_trees)


def pickle_dump_native_forest(native_forest, filename):
    f = open(filename, "wb")
    pyforest = as_pyforest(native_forest)
    pickle.dump(pyforest, f)


def pickle_load_native_forest(filename):
    f = open(filename, "rb")
    pyforest = pickle.load(f)
    native_forest = as_nativeforest(pyforest)
    return native_forest