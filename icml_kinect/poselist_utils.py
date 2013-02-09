import numpy as np
import matplotlib.pyplot as pl
import cPickle as pickle
import argparse
import os

import utils as kinect_utils

def images_to_pickles(foldername, poses_file):
    poses_to_include_file = open(poses_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    for i, pose_filename in enumerate(pose_filenames):
        print "Processing %d - %s" % (i, pose_filename)

        depth_pickle_file = "%s%s.exr" % (foldername, pose_filename)
        if not os.path.exists(depth_pickle_file):
            if os.path.exists("%s%s.exr" % (foldername, pose_filename)):
              depths = np.clip(kinect_utils.load_depth("%s%s.exr" % (foldername, pose_filename)), 0.0, 6.0)
              pickle.dump(depths, open(depth_pickle_file, 'wb'))

        class_labels_pickle_file = "%s%s_classlabels.pkl" % (foldername, pose_filename)
        if not os.path.exists(class_labels_pickle_file):
            if os.path.exists("%s%s.png" % (foldername, pose_filename)):
              class_labels_png = pl.imread("%s%s.png" % (foldername, pose_filename))
              class_labels = kinect_utils.image_labels(class_labels_png)
              pickle.dump(class_labels, open(class_labels_pickle_file, 'wb'))

def merge_poses(poses_to_include_file_list, poses_to_ignore_file_list):
    ignore_pose_filenames = []
    for ignore_poses_file in poses_to_ignore_file_list:
        poses_to_ignore_file = open(ignore_poses_file, 'r')
        pose_filenames = poses_to_ignore_file.read().split('\n')
        ignore_pose_filenames.extend(pose_filenames)

    output_pose_filenames = []
    for poses_file in poses_to_include_file_list:
        poses_to_include_file = open(poses_file, 'r')
        pose_filenames = poses_to_include_file.read().split('\n')

        for pose_filename in pose_filenames:
            if pose_filename not in output_pose_filenames and pose_filename not in ignore_pose_filenames:
                output_pose_filenames.append(pose_filename)
    print '\n'.join(output_pose_filenames)

