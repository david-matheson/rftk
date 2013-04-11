import numpy as np

import cPickle as pickle
import gzip
import argparse

import utils as kinect_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-m', '--number_of_pixels_per_image', type=int, required=True)
    parser.add_argument('-os', '--output_sampled_pickle_file', type=str, required=True)
    parser.add_argument('-or', '--output_raw_pickle_file', type=str, required=True)
    args = parser.parse_args()

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    depths_buffer, pixel_indices_buffer, pixel_labels_buffer = kinect_utils.load_data_and_sample(args.pose_files_input_path,
                                                                        pose_filenames[0:args.number_of_images],
                                                                        args.number_of_pixels_per_image)
    pickle.dump((depths_buffer, pixel_indices_buffer, pixel_labels_buffer), gzip.open(args.output_sampled_pickle_file, 'wb'))


    depths_buffer, labels_buffer = kinect_utils.load_data(args.pose_files_input_path,
                                            pose_filenames[0:args.number_of_images])
    pickle.dump((depths_buffer, labels_buffer), gzip.open(args.output_raw_pickle_file, 'wb'))



