import numpy as np
import matplotlib.pyplot as pl
import cPickle as pickle
from datetime import datetime
import argparse
import os

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data as forest_data
import rftk.native.features
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train

import rftk.utils.buffer_converters as buffer_converters
import rftk.utils.forest as forest_utils

import utils as kinect_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Highlight words')
    parser.add_argument('-i', '--ground_truth_in_path', type=str, required=True)
    parser.add_argument('-f', '--forest_input_path', type=str, required=True)
    parser.add_argument('-o', '--out_path', type=str, required=True)
    args = parser.parse_args()

    depths = pickle.load(open(args.ground_truth_in_path+"depths.pkl",'rb'))
    labels = pickle.load(open(args.ground_truth_in_path+"labels.pkl",'rb'))
    (numberOfImages, depthM, depthN) = depths.shape

    # Load forest
    forest_id = 2
    forest_pickle_filename = "%s/forest-%d.pkl" % (args.forest_input_path, forest_id)
    forest = forest_utils.pickle_load_native_forest(forest_pickle_filename)

    kinect_utils.plot_classification_imgs(args.out_path, depths, labels, forest)
    accuracy = kinect_utils.classification_accuracy(depths, labels, forest)
    print accuracy