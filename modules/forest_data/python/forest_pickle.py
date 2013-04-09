import cPickle as pickle
import rftk.buffers as buffers

# This is legacy (remove once it is no longer referenced)
def pickle_dump_native_forest(native_forest, filename):
    f = open(filename, "wb")
    pickle.dump(native_forest, f)

# This is legacy (remove once it is no longer referenced)
def pickle_load_native_forest(filename):
    f = open(filename, "rb")
    native_forest = pickle.load(f)
    return native_forest


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