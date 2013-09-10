import numpy as np
import re

def parse_ecoli_record(record):
    fields = re.split("\s+", record.strip())
    # The first field is a name (non-predictive) and the last field is the label
    x, y = fields[1:-1], fields[-1]
    x = map(float, x)
    return x, y

def load_ecoli_data():
    with open("tests/data/ecoli.data") as data_file:
        data = data_file.readlines()

    X, Y = zip(*map(parse_ecoli_record, data))
    class_names = list(set(Y))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray([ class_names.index(y) for y in Y ], dtype=np.int32)

    numberOfSampled = len(Y);
    perm = np.random.permutation(numberOfSampled)
    X = X[perm]
    Y = Y[perm]
    trainSize = int(numberOfSampled*0.6)
    return X[0:trainSize], Y[0:trainSize], X[trainSize:-1], Y[trainSize:-1]


def load_sklearn_data(file):
    import sklearn.datasets as sklearn_datasets
    data = sklearn_datasets.load_svmlight_file(file)
    X = np.array(data[0].todense(),  dtype=np.float32)
    Y = np.array( data[1], dtype=np.int32 )
    return X, Y

def load_usps_data():
    (x_train, y_train) = load_sklearn_data("tests/data/usps")
    (x_test, y_test) = load_sklearn_data("tests/data/usps.t")
    return x_train, y_train, x_test, y_test

def parse_wine_record(record):
    fields = re.split(";", record.strip())
    # the last field is the target
    x, y = fields[:-1], fields[-1]
    x = map(float, x)
    y = float(y)
    return x, y

def load_wine_data(normalize_data=False):
    with open("tests/data/winequality-red.csv") as data_file:
        data = data_file.readlines()

    # The first line is a header
    X, Y = zip(*map(parse_wine_record, data[1:]))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32).reshape((-1,1))

    if normalize_data:
      x_abs_max = np.max(np.abs(X), axis=0)
      X /= x_abs_max

    numberOfSampled = len(Y);
    perm = np.random.permutation(numberOfSampled)
    X = X[perm]
    Y = Y[perm]
    trainSize = int(numberOfSampled*0.75)
    return X[0:trainSize], Y[0:trainSize], X[trainSize:-1], Y[trainSize:-1]

def load_kinect_data(numpy_filename):
    f = open(numpy_filename, 'rb')
    depths = np.load(f)
    labels = np.load(f)
    joints = np.load(f)
    pixel_indices = np.load(f)
    pixel_labels = np.load(f)
    joint_offsets = np.array(np.load(f), dtype=np.float32)
    return depths, labels, pixel_indices, pixel_labels, joint_offsets

def load_kinect_train_data():
    return load_kinect_data("tests/data/train_bundle_n10_m100_320x240.np")

def load_kinect_test_data():
    return load_kinect_data("tests/data/test_bundle_n10_m100_320x240.np")