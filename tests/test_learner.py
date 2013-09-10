import unittest as unittest
import numpy as np

import load_data 

import rftk



class TestNew(unittest.TestCase):

    def test_vanilia_classifier(self):
        learner = rftk.learn.create_vanilia_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes, bootstrap=False, number_of_features=2)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 2)

    def test_vanilia_classifier_breadth_first(self):
        learner = rftk.learn.create_vanilia_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes, bootstrap=False, tree_order='breadth_first', number_of_features=2)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 2)

    def test_one_stream_classifier(self):
        learner = rftk.learn.create_one_stream_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes,
                                bootstrap=False,
                                number_of_features=2,
                                number_of_splitpoints=10,
                                min_child_size=1)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 2)

    def test_two_stream_classifier_balanced(self):
        learner = rftk.learn.create_two_stream_classifier()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0,1], [0,2], [0,2]], dtype=np.float32)
        classes = np.array([0,0,0,1,1,2,2], dtype=np.int32)
        predictor = learner.fit(x=x, classes=classes,
                                bootstrap=False,
                                number_of_trees=100,
                                number_of_features=2,
                                number_of_splitpoints=10,
                                probability_of_impurity_stream=0.5,
                                min_child_size=1)
        result_prob = predictor.predict(x=x)
        result = predictor.predict(x=x).argmax(axis=1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 1)
        self.assertEqual(result[5], 2)
        self.assertEqual(result[6], 2)

    def test_standard_regression(self):
        learner = rftk.learn.create_standard_regression()
        x = np.array([[3,1],[3,2], [3,3], [0,1], [0.5,1.5], [1,1.8], [1.5,2]], dtype=np.float32)
        y = np.array([[0,0],[0,0],[0,0],[1,3],[1,3],[2,-1],[2,-1]], dtype=np.float32)
        predictor = learner.fit(x=x, y=y, bootstrap=False, number_of_features=2, number_of_trees=10, number_of_leaves=10000000)
        result = predictor.predict(x=x)
        self.assertEqual(result[0,0], 0)
        self.assertEqual(result[0,1], 0)
        self.assertEqual(result[1,0], 0)
        self.assertEqual(result[1,1], 0)
        self.assertEqual(result[2,0], 0)
        self.assertEqual(result[2,1], 0)
        self.assertEqual(result[3,0], 1)
        self.assertEqual(result[3,1], 3)
        self.assertEqual(result[4,0], 1)
        self.assertEqual(result[4,1], 3)
        self.assertEqual(result[5,0], 2)
        self.assertEqual(result[5,1], -1)
        self.assertEqual(result[5,0], 2)
        self.assertEqual(result[5,1], -1)

    def test_ecoli_classifiers(self):
        x_train, y_train, x_test, y_test = load_data.load_ecoli_data()
        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=False, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train, tree_weights=rftk.buffers.as_vector_buffer( np.ones(predictor.get_forest().GetNumberOfTrees(), dtype=np.float64))).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_vanilia_classifier %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_vanilia_classifier bootstrap %f %f %f" % (acc_train, acc_train_oob, acc_test))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, tree_order='breadth_first', number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_vanilia_classifier tree_order==breadth_first %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)

        # forest = predictor.get_forest()
        # tree = forest.GetTree(0)
        # extraInfo = tree.mExtraInfo
        # extraInfo.Print()

        learner = rftk.learn.create_dimension_pair_difference_matrix_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=False, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_dimension_pair_difference_matrix_classifier %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_class_pair_difference_matrix_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=False, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_class_pair_difference_matrix_classifier %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_one_stream_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=False, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_one_stream_classifier %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_two_stream_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=False, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_two_stream_classifier %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_online_one_stream_classifier()
        for i in range(3):
            predictor = learner.fit(x=x_train, classes=y_train, number_of_trees=20, number_of_jobs=5,
                                    number_of_splitpoints=100,
                                    min_impurity=0.1, 
                                    bootstrap=True,
                                    min_child_size_sum=3,
                                    max_frontier_size=50000 )
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_online_one_stream_classifier %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_online_two_stream_consistent_classifier()
        for i in range(3):
            predictor = learner.fit(x=x_train, classes=y_train, number_of_trees=20, number_of_jobs=5,
                                        number_of_splitpoints=100,
                                        min_impurity=0.1, 
                                        poisson_sample=1.0, 
                                        number_of_data_to_split_root=3, 
                                        number_of_data_to_force_split_root=20, 
                                        split_rate_growth=1.001,
                                        probability_of_impurity_stream=0.5,
                                        max_frontier_size=50000 )
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("create_online_two_stream_consistent_classifier %f %f" % (acc_train, acc_test))
        self.assertGreater(acc_test, 0.7)


    def test_wine_regression(self):
        x_train, y_train, x_test, y_test = load_data.load_wine_data()

        learner = rftk.learn.create_standard_regression()
        predictor = learner.fit(x=x_train, y=y_train, bootstrap=False, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train)
        mse_train = np.mean((y_train - y_hat_train)**2)
        y_hat_test = predictor.predict(x=x_test)
        mse_test = np.mean((y_test - y_hat_test)**2)
        print("create_standard_regression %f %f" % (mse_train, mse_test))
        self.assertLess(mse_test, 0.5)

        learner = rftk.learn.create_standard_regression()
        predictor = learner.fit(x=x_train, y=y_train, bootstrap=True, number_of_trees=100, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train)
        mse_train = np.mean((y_train - y_hat_train)**2)
        y_hat_train_oob = predictor.predict_oob(x=x_train)
        mse_train_oob = np.mean((y_train - y_hat_train_oob)**2)
        y_hat_test = predictor.predict(x=x_test)
        mse_test = np.mean((y_test - y_hat_test)**2)
        print("create_standard_regression bootstrap %f %f %f" % (mse_train, mse_train_oob, mse_test))
        self.assertLess(mse_test, 0.5)

        learner = rftk.learn.create_biau2008_regression()
        predictor = learner.fit(x=x_train, y=y_train, number_of_trees=20, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train)
        mse_train = np.mean((y_train - y_hat_train)**2)
        y_hat_test = predictor.predict(x=x_test)
        mse_test = np.mean((y_test - y_hat_test)**2)
        print("create_biau2008_regression %f %f" % (mse_train, mse_test))
        self.assertLess(mse_test, 0.5)

        learner = rftk.learn.create_consistent_regression()
        predictor = learner.fit(x=x_train, y=y_train, number_of_trees=20, number_of_jobs=1)
        y_hat_train = predictor.predict(x=x_train)
        mse_train = np.mean((y_train - y_hat_train)**2)
        y_hat_test = predictor.predict(x=x_test)
        mse_test = np.mean((y_test - y_hat_test)**2)
        print("create_consistent_regression %f %f" % (mse_train, mse_test))
        self.assertLess(mse_test, 0.6)

        x_train, y_train, x_test, y_test = load_data.load_wine_data(normalize_data=True)
        learner = rftk.learn.create_biau2012_regression()
        predictor = learner.fit(x=x_train, y=y_train, number_of_trees=20, number_of_jobs=1, min_impurity=-1, min_node_size=-1)
        y_hat_train = predictor.predict(x=x_train)
        mse_train = np.mean((y_train - y_hat_train)**2)
        y_hat_test = predictor.predict(x=x_test)
        mse_test = np.mean((y_test - y_hat_test)**2)

        # forest = predictor.get_forest()
        # tree = forest.GetTree(0)
        # extraInfo = tree.mExtraInfo
        # extraInfo.Print()

        print("create_biau2012_regression %f %f" % (mse_train, mse_test))
        self.assertLess(mse_test, 0.8)


    def test_vanilia_scaled_depth_delta_classifier(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        
        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        # datapoint_indices = rftk.buffers.as_vector_buffer(np.array(np.arange(number_of_datapoints), dtype=np.int32))

        forest_learner = rftk.learn.create_vanilia_scaled_depth_delta_classifier(
                                      number_of_trees=5,
                                      number_of_features=2000,
                                      min_impurity_gain=0.1,
                                      # max_depth=30,
                                      min_samples_split=5,
                                      # min_samples_leaf = 5,
                                      # min_impurity_gain = 0.01
                                      ux=75, uy=75, vx=75, vy=75,
                                      bootstrap=True,
                                      number_of_jobs=5)

        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_pixel_label_buffer = rftk.buffers.as_vector_buffer(train_pixel_labels)
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      # pixel_indices=train_pixel_indices_buffer.Slice(datapoint_indices), 
                                      offset_scales=offset_scales_buffer,
                                      classes=train_pixel_label_buffer
                                      # classes=train_pixel_label_buffer.Slice(datapoint_indices)
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_accurracy = np.mean(train_pixel_labels == train_predictions.argmax(axis=1))
        train_predictions_oob = predictor.predict_oob(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_accurracy_oob = np.mean(train_pixel_labels == train_predictions_oob.argmax(axis=1))
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_accurracy = np.mean(test_pixel_labels == test_predictions.argmax(axis=1))
        print("create_vanilia_scaled_depth_delta_classifier %f %f %f" % (train_accurracy, train_accurracy_oob, test_accurracy))


    def test_online_one_stream_depth_delta_classifier(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        
        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        forest_learner = rftk.learn.create_online_one_stream_depth_delta_classifier(
                                      number_of_trees=5,
                                      number_of_features=2000,
                                      min_impurity_gain=0.1,
                                      # max_depth=30,
                                      min_samples_split=5,
                                      # min_samples_leaf = 5,
                                      # min_impurity_gain = 0.01
                                      ux=75, uy=75, vx=75, vy=75,
                                      bootstrap=True,
                                      number_of_jobs=5)

        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_pixel_label_buffer = rftk.buffers.as_vector_buffer(train_pixel_labels)
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      # pixel_indices=train_pixel_indices_buffer.Slice(datapoint_indices), 
                                      offset_scales=offset_scales_buffer,
                                      classes=train_pixel_label_buffer
                                      # classes=train_pixel_label_buffer.Slice(datapoint_indices)
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_accurracy = np.mean(train_pixel_labels == train_predictions.argmax(axis=1))
        train_predictions_oob = predictor.predict_oob(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_accurracy_oob = np.mean(train_pixel_labels == train_predictions_oob.argmax(axis=1))
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_accurracy = np.mean(test_pixel_labels == test_predictions.argmax(axis=1))
        print("create_online_one_stream_depth_delta_classifier %f %f %f" % (train_accurracy, train_accurracy_oob, test_accurracy))


    def test_online_two_stream_consistent_depth_delta_classifier(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        
        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        forest_learner = rftk.learn.create_online_two_stream_consistent_depth_delta_classifier(
                                      number_of_trees=5,
                                      number_of_features=2000,
                                      min_impurity=0.1,
                                      # max_depth=30,
                                      number_of_data_to_split_root=5,
                                      number_of_data_to_force_split_root=10,
                                      split_rate_growth=1.001,
                                      # min_samples_leaf = 5,
                                      # min_impurity_gain = 0.01
                                      ux=75, uy=75, vx=75, vy=75,
                                      bootstrap=True,
                                      number_of_jobs=5)

        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_pixel_label_buffer = rftk.buffers.as_vector_buffer(train_pixel_labels)
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      # pixel_indices=train_pixel_indices_buffer.Slice(datapoint_indices), 
                                      offset_scales=offset_scales_buffer,
                                      classes=train_pixel_label_buffer
                                      # classes=train_pixel_label_buffer.Slice(datapoint_indices)
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_accurracy = np.mean(train_pixel_labels == train_predictions.argmax(axis=1))
        train_predictions_oob = predictor.predict_oob(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_accurracy_oob = np.mean(train_pixel_labels == train_predictions_oob.argmax(axis=1))
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_accurracy = np.mean(test_pixel_labels == test_predictions.argmax(axis=1))
        print("create_online_two_stream_consistent_depth_delta_classifier %f %f %f" % (train_accurracy, train_accurracy_oob, test_accurracy))



    def test_vanilia_scaled_depth_delta_regression(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        number_of_datapoints = len(train_labels)

        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        forest_learner = rftk.learn.create_vanilia_scaled_depth_delta_regression(
                                    number_of_trees=5,
                                    number_of_features=1000,
                                    min_node_size=10,
                                    min_child_size = 5,
                                    ux=40, uy=40, vx=40, vy=40,
                                    min_impurity=0.0,
                                    bootstrap=True,
                                    number_of_jobs=5)

        joint_id = 5
        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_joint_offsets_buffer = rftk.buffers.as_matrix_buffer(train_joint_offsets[:, joint_id, :])
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      y=train_joint_offsets_buffer
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_mse = np.mean((train_joint_offsets[:, joint_id, :] - train_predictions)**2)
        
        train_predictions_oob = predictor.predict_oob(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_mse_oob = np.mean((train_joint_offsets[:, joint_id, :] - train_predictions_oob)**2)
        
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_mse = np.mean((test_joint_offsets[:, joint_id, :] - test_predictions)**2)
        
        print("create_vanilia_scaled_depth_delta_regression %f %f %f" % (train_mse, train_mse_oob, test_mse))



    def test_biau2008_scaled_depth_delta_regression(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        number_of_datapoints = len(train_labels)

        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        forest_learner = rftk.learn.create_biau2008_scaled_depth_delta_regression(
                                    number_of_trees=5,
                                    number_of_split_retries = 1000,
                                    # ux=75, uy=75, vx=75, vy=75,
                                    ux=40, uy=40, vx=40, vy=40,
                                    bootstrap=False,
                                    number_of_jobs=5)

        joint_id = 5
        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_joint_offsets_buffer = rftk.buffers.as_matrix_buffer(train_joint_offsets[:, joint_id, :])
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      y=train_joint_offsets_buffer
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_mse = np.mean((train_joint_offsets[:, joint_id, :] - train_predictions)**2)
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_mse = np.mean((test_joint_offsets[:, joint_id, :] - test_predictions)**2)
        print("create_biau2008_scaled_depth_delta_regression %f %f" % (train_mse, test_mse))


    def test_biau2008_scaled_one_split_depth_delta_regression(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        number_of_datapoints = len(train_labels)

        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        forest_learner = rftk.learn.create_biau2008_scaled_depth_delta_regression(
                                    number_of_trees=5,
                                    number_of_split_retries = 1000,
                                    # ux=75, uy=75, vx=75, vy=75,
                                    ux=40, uy=40, vx=40, vy=40,
                                    bootstrap=False,
                                    number_of_leaves=1,
                                    number_of_jobs=5)

        joint_id = 5
        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_joint_offsets_buffer = rftk.buffers.as_matrix_buffer(train_joint_offsets[:, joint_id, :])
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      y=train_joint_offsets_buffer
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_mse = np.mean((train_joint_offsets[:, joint_id, :] - train_predictions)**2)
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_mse = np.mean((test_joint_offsets[:, joint_id, :] - test_predictions)**2)
        print("create_biau2008_scaled_one_split_depth_delta_regression %f %f" % (train_mse, test_mse))



    def test_biau2012_scaled_depth_delta_regression(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        number_of_datapoints = len(train_labels)

        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        forest_learner = rftk.learn.create_biau2012_scaled_depth_delta_regression(
                                    number_of_trees=5,
                                    number_of_features=2000,
                                    min_node_size=1,
                                    # min_child_size = 5,
                                    # ux=75, uy=75, vx=75, vy=75,
                                    ux=40, uy=40, vx=40, vy=40,
                                    # min_impurity=0.0,
                                    # bootstrap=False,
                                    number_of_jobs=5)

        joint_id = 5
        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_joint_offsets_buffer = rftk.buffers.as_matrix_buffer(train_joint_offsets[:, joint_id, :])
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      y=train_joint_offsets_buffer
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_mse = np.mean((train_joint_offsets[:, joint_id, :] - train_predictions)**2)
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_mse = np.mean((test_joint_offsets[:, joint_id, :] - test_predictions)**2)
        print("create_biau2012_scaled_depth_delta_regression %f %f" % (train_mse, test_mse))



    def test_consistent_scaled_depth_delta_regression(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        number_of_datapoints = len(train_labels)

        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        forest_learner = rftk.learn.create_consistent_scaled_depth_delta_regression(
                                    number_of_trees=5,
                                    number_of_features=2000,
                                    min_node_size=10,
                                    min_child_size = 5,
                                    # ux=75, uy=75, vx=75, vy=75,
                                    ux=40, uy=40, vx=40, vy=40,
                                    min_impurity=0.0,
                                    # bootstrap=False,
                                    poisson_number_of_features=True,
                                    number_of_jobs=5)

        joint_id = 5
        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_joint_offsets_buffer = rftk.buffers.as_matrix_buffer(train_joint_offsets[:, joint_id, :])
        predictor = forest_learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      y=train_joint_offsets_buffer
                                      )

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_mse = np.mean((train_joint_offsets[:, joint_id, :] - train_predictions)**2)
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_mse = np.mean((test_joint_offsets[:, joint_id, :] - test_predictions)**2)
        print("create_consistent_scaled_depth_delta_regression %f %f" % (train_mse, test_mse))


    def test_ecoli_greedy_classifiers(self):
        x_train, y_train, x_test, y_test = load_data.load_ecoli_data()
        print x_train.shape

        learner = rftk.learn.create_greedy_add_swap_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=2, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=3, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=1, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=3, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=2, number_of_trees=100, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("ecoli create_greedy_add_swap_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_greedy_add_swap_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=2, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=3, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=1, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=3, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=2, number_of_trees=100, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("ecoli create_greedy_add_swap_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_greedy_add_swap_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=2, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=3, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=1, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=3, number_of_trees=100, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_features=2, number_of_trees=100, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("ecoli create_greedy_add_swap_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=500, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("ecoli create_vanilia_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=500, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("ecoli create_vanilia_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=500, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("ecoli create_vanilia_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=500, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("ecoli create_vanilia_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

    def test_usps_greedy_classifiers(self):
        x_train, y_train, x_test, y_test = load_data.load_usps_data()
        print x_train.shape
        print x_test.shape

        learner = rftk.learn.create_greedy_add_swap_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=10, number_of_jobs=5)
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=40, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("usps create_greedy_add_swap_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)

        learner = rftk.learn.create_vanilia_classifier()
        predictor = learner.fit(x=x_train, classes=y_train, bootstrap=True, number_of_trees=50, number_of_jobs=5)
        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
        acc_train_oob = np.mean(y_train == y_hat_train_oob)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)
        print("usps create_vanilia_classifier bootstrap %f %f %f (#trees=%d)" % (acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        self.assertGreater(acc_test, 0.7)


if __name__ == '__main__':
    unittest.main()
