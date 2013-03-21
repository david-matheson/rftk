import cPickle as pickle
import numpy as np

import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.forest_data as forest_data


class PyTree(object):
    pass

def as_pyforest(native_forest):
    pytrees = []
    for tree_id in range(native_forest.GetNumberOfTrees()):
        native_tree = native_forest.GetTree(tree_id)
        native_tree.Compact()
        pytree = PyTree()
        pytree.path = buffers.as_numpy_array(native_tree.mPath)
        pytree.int_features_params = buffers.as_numpy_array(native_tree.mIntFeatureParams)
        pytree.float_features_params = buffers.as_numpy_array(native_tree.mFloatFeatureParams)
        pytree.depths = buffers.as_numpy_array(native_tree.mDepths, flatten=True)
        pytree.counts = buffers.as_numpy_array(native_tree.mCounts, flatten=True)
        pytree.ys = buffers.as_numpy_array(native_tree.mYs)
        pytrees.append(pytree)
    return pytrees

def as_nativeforest(pytrees):
    native_trees = []
    for tree in pytrees:
        native_tree = forest_data.Tree(buffers.as_matrix_buffer(tree.path),
                                        buffers.as_matrix_buffer(tree.int_features_params),
                                        buffers.as_matrix_buffer(tree.float_features_params),
                                        buffers.as_vector_buffer(tree.depths),
                                        buffers.as_vector_buffer(tree.counts),
                                        buffers.as_matrix_buffer(tree.ys))
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