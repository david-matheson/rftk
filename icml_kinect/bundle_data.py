import numpy as np

import cPickle as pickle
import gzip
import argparse

import rftk.buffers as buffers
import utils as kinect_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-m', '--number_of_pixels_per_image', type=int, required=True)
    parser.add_argument('-o', '--output_pickle_file', type=str, required=True)
    args = parser.parse_args()

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    depths_buffer, labels_buffer, pixel_indices_buffer, pixel_labels_buffer = kinect_utils.load_data_and_sample(args.pose_files_input_path,
                                                                        pose_filenames[0:args.number_of_images],
                                                                        args.number_of_pixels_per_image)
    depths = buffers.as_numpy_array(depths_buffer)
    labels = buffers.as_numpy_array(labels_buffer)
    pixel_indices = buffers.as_numpy_array(pixel_indices_buffer)
    pixel_labels = buffers.as_numpy_array(pixel_labels_buffer)

    f = open(args.output_pickle_file, 'wb')
    np.save(f, depths)
    np.save(f, labels)
    np.save(f, pixel_indices)
    np.save(f, pixel_labels)
    
    # f = gzip.open(args.output_pickle_file, 'wb')
    # pickle.dump(depths, f)
    # pickle.dump(labels, f)
    # pickle.dump(pixel_indices, f)
    # pickle.dump(pixel_labels, f)

