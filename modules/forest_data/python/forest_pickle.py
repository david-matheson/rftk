import cPickle as pickle

# This is legacy (remove once it is no longer referenced)
def pickle_dump_native_forest(native_forest, filename):
    f = open(filename, "wb")
    pickle.dump(native_forest, f)

# This is legacy (remove once it is no longer referenced)
def pickle_load_native_forest(filename):
    f = open(filename, "rb")
    native_forest = pickle.load(f)
    return native_forest