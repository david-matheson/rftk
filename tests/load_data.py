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



def parse_wine_record(record):
    fields = re.split(";", record.strip())
    # the last field is the target
    x, y = fields[:-1], fields[-1]
    x = map(float, x)
    y = float(y)
    return x, y

def load_wine_data():
    with open("tests/data/winequality-red.csv") as data_file:
        data = data_file.readlines()

    # The first line is a header
    X, Y = zip(*map(parse_wine_record, data[1:]))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32).reshape((-1,1))

    numberOfSampled = len(Y);
    perm = np.random.permutation(numberOfSampled)
    X = X[perm]
    Y = Y[perm]
    trainSize = int(numberOfSampled*0.75)
    return X[0:trainSize], Y[0:trainSize], X[trainSize:-1], Y[trainSize:-1]

