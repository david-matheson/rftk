#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Offline training of kinect random forests
'''

import numpy as np
import cPickle as pickle
import gzip
from datetime import datetime
import argparse
import os

import rftk

import utils as kinect_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    args = parser.parse_args()

    offline_run_folder = ("experiment_data_offline/offline-tree-%d-n-%d-%s-standard") % (
                            args.number_of_trees,
                            args.number_of_images,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(offline_run_folder):
        os.makedirs(offline_run_folder)

    number_of_pixels_per_image = 1000
    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    depths_buffer, pixel_indices_buffer, pixel_labels_buffer = kinect_utils.load_data_and_sample(args.pose_files_input_path,
                                                                        pose_filenames[0:args.number_of_images],
                                                                        number_of_pixels_per_image)

    # Randomly offset scales
    number_of_datapoints = pixel_indices_buffer.GetM()
    offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)
    offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

    forest_learner = rftk.learn.create_vanilia_scaled_depth_delta_classifier()
    predictor = forest_learner.fit(depth_images=depths_buffer, 
                                  pixel_indices=pixel_indices_buffer, 
                                  offset_scales=offset_scales_buffer,
                                  classes=pixel_labels_buffer,
                                  number_of_trees=args.number_of_trees,
                                  number_of_features=5,
                                  max_depth=30,
                                  min_samples_split=10,
                                  # min_samples_leaf = 5,
                                  # min_impurity_gain = 0.01
                                  ux=75, uy=75, vx=75, vy=75,
                                  bootstrap=True,
                                  number_of_jobs=2)

    predictions = predictor.predict(depth_images=depths_buffer,pixel_indices=pixel_indices_buffer)
    accurracy = np.mean(rftk.buffers.as_numpy_array(pixel_labels_buffer) == predictions.argmax(axis=1))
    print accurracy

    forest = predictor.get_forest()

    # Print forest stats
    forestStats = forest.GetForestStats()
    print forest.GetNumberOfTrees()
    forestStats.Print()

    #pickle forest and data used for training
    forest_pickle_filename = "%s/forest-1-%d.pkl" % (offline_run_folder, args.number_of_images)
    pickle.dump(forest, gzip.open(forest_pickle_filename, 'wb'))
