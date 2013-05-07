import argparse
import numpy as np
import gzip
import cPickle as pickle
import glob
import os

import kinect_utils as kinect_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deterime forest accuracy on test set')
    parser.add_argument('-p', '--test_poses', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-f', '--forest_paths', type=str, required=True)
    args = parser.parse_args()

    f = open(args.test_poses, 'rb')
    depths = np.load(f)
    labels = np.load(f)

    directory = glob.glob(args.forest_paths)
    for d in directory:
        files = glob.glob('%s/forest*.pkl' % d)
        for f in files:
            accuracy_filename = f.replace("forest", "accuracy")
            if not os.path.exists(accuracy_filename):
              print accuracy_filename
              forest = pickle.load(gzip.open(f, 'rb'))
              forest_stats = forest.GetForestStats()
              forest_stats.Print()
              accuracy = kinect_utils.classification_accuracy(depths[0:args.number_of_images,:,:],
                                                              labels[0:args.number_of_images,:,:],
                                                              forest)
              print accuracy
              pickle.dump(accuracy, file(accuracy_filename, 'wb'))
