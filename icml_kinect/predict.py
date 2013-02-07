import numpy as np
import pickle

import pixel_rf.tree.loader as treeloader
import pixel_rf.treeutils.treeutils as treeutils

# def load_class_forest(path, prefix, tree_ids):
#     # Load trees from pickle files
#     trees = [treeloader.Tree(path, prefix, index) for index in tree_ids]

#     # Flatten to 3d numpy arrays
#     paths = treeloader.listToNp([tree.treePath for tree in trees])
#     int_params = treeloader.listToNp([tree.treeIntParams  for tree in trees])
#     float_params = treeloader.listToNp([tree.treeFloatParams  for tree in trees])
#     ys = treeloader.listToNp([tree.treeYs  for tree in trees])

#     pixel_class_forest = treeutils.PixelClassForest(treeutils.Forest(paths, int_params, float_params, ys))
#     return pixel_class_forest


def classifyPixels(depth_data, pixel_class_forest):
    # Create output buffers
    (depthM, depthN) = depth_data.shape
    labels = np.zeros((depthM, depthN), dtype=np.uint8)
    per_pixel_probabilities = np.zeros((depthM, depthN), dtype=np.float32)
    depth_data_buffer = treeutils.FloatImgBuffer(depth_data)

    pixel_class_forest.PredictBreadthFirst(depth_data_buffer, labels, per_pixel_probabilities)
    return (labels, per_pixel_probabilities)


# def load_offset_forests(number_of_bodyparts, number_of_joints, min_probability, max_depth,
#                         path, prefix_format, bodypart_and_joints, tree_ids):

#     offset_forests = treeutils.PixelJointOffsetForest(number_of_bodyparts, number_of_joints, min_probability, max_depth)

#     for (bodypart, joint) in bodypart_and_joints:
#         prefix = prefix_format % (bodypart,joint)
#         offset_trees = [treeloader.Tree(path, prefix, index) for index in tree_ids]

#         paths = treeloader.listToNp([tree.treePath for tree in offset_trees])
#         int_params = treeloader.listToNp([tree.treeIntParams  for tree in offset_trees])
#         float_params = treeloader.listToNp([tree.treeFloatParams  for tree in offset_trees])
#         tests = treeloader.listToNp([tree.treeFloatParams  for tree in offset_trees])
#         ys = treeloader.listToNp([tree.treeYs  for tree in offset_trees])

#         offset_forests.AddForest(bodypart, joint, paths, int_params, float_params, ys)

#     return offset_forests


# def predict_joint_offsets(depth_data, pixel_class_forest, offset_forests, joints_dim):
#     (depthM, depthN) = depth_data.shape
#     labels = np.zeros((depthM, depthN), dtype=np.uint8)
#     per_pixel_probabilities = np.zeros((depthM, depthN), dtype=np.float32)
#     depth_data_buffer = treeutils.FloatImgBuffer(depth_data)
#     pixel_class_forest.PredictBreadthFirst(depth_data_buffer, labels, per_pixel_probabilities)

#     number_of_joints = offset_forests.GetNumJoints()
#     votes_by = np.zeros((number_of_joints, depthM, depthN), dtype=np.float32)
#     votes_for = np.zeros((number_of_joints, depthM, depthN), dtype=np.float32)
#     joint_centers = np.zeros((number_of_joints, joints_dim), dtype=np.float32)
#     offset_forests.Predict(depth_data, labels, per_pixel_probabilities, votes_by, votes_for, joint_centers)

#     return (labels, per_pixel_probabilities, votes_by, votes_for, joint_centers)


