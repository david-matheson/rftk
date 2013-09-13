import unittest as unittest
import numpy as np
import datetime

import load_data 

import rftk

def run_classifier(learner, description, 
                            x_train, y_train, x_test, y_test, 
                            number_of_trees_list, 
                            bootstrap=False,
                            number_of_features_list=None,
                            number_of_jobs=5):

        for i, number_of_trees in enumerate(number_of_trees_list):
            if number_of_features_list is not None:
                predictor = learner.fit(x=x_train, classes=y_train, bootstrap=bootstrap, number_of_trees=number_of_trees, number_of_features=number_of_features_list[i], number_of_jobs=number_of_jobs)
            else:
                predictor = learner.fit(x=x_train, classes=y_train, bootstrap=bootstrap, number_of_trees=number_of_trees, number_of_jobs=number_of_jobs)


        y_hat_train = predictor.predict(x=x_train).argmax(axis=1)
        acc_train = np.mean(y_train == y_hat_train)
        y_hat_test = predictor.predict(x=x_test).argmax(axis=1)
        acc_test = np.mean(y_test == y_hat_test)

        if bootstrap:
            y_hat_train_oob = predictor.predict_oob(x=x_train).argmax(axis=1)
            acc_train_oob = np.mean(y_train == y_hat_train_oob)
            print("%s %f %f %f (#trees=%d)" % (description, acc_train, acc_train_oob, acc_test, predictor.get_forest().GetNumberOfTrees()))
        else:
            print("%s %f %f (#trees=%d)" % (description, acc_train, acc_test, predictor.get_forest().GetNumberOfTrees()))

        return acc_test



def run_regression(learner, description, 
                            x_train, y_train, x_test, y_test, 
                            number_of_trees_list, 
                            bootstrap=False,
                            number_of_features_list=None,
                            number_of_jobs=5):

        for i, number_of_trees in enumerate(number_of_trees_list):
            if number_of_features_list is not None:
                predictor = learner.fit(x=x_train, y=y_train, bootstrap=bootstrap, number_of_trees=number_of_trees, number_of_features=number_of_features_list[i], number_of_jobs=number_of_jobs)
            else:
                predictor = learner.fit(x=x_train, y=y_train, bootstrap=bootstrap, number_of_trees=number_of_trees, number_of_jobs=number_of_jobs)


        y_hat_train = predictor.predict(x=x_train)
        mse_train = np.mean((y_train - y_hat_train)**2)
        y_hat_test = predictor.predict(x=x_test)
        mse_test = np.mean((y_test - y_hat_test)**2)

        if bootstrap:
            y_hat_train_oob = predictor.predict_oob(x=x_train)
            mse_train_oob = np.mean((y_train - y_hat_train_oob)**2)
            print("%s %f %f %f (#trees=%d)" % (description, mse_train, mse_train_oob, mse_test, predictor.get_forest().GetNumberOfTrees()))
        else:
            print("%s %f %f (#trees=%d)" % (description, mse_train, mse_test, predictor.get_forest().GetNumberOfTrees()))

        return mse_train

def run_depth_delta_classifier(learner, description, 
                            train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets,
                            test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets, 
                            number_of_trees_list, 
                            bootstrap=False,
                            number_of_features_list=None,
                            number_of_jobs=5):

        number_of_datapoints = len(train_pixel_labels)
        offset_scales = np.array(np.random.uniform(0.99, 1.0, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

        train_depths_buffer = rftk.buffers.as_tensor_buffer(train_depths)
        train_pixel_indices_buffer = rftk.buffers.as_matrix_buffer(train_pixel_indices)
        train_pixel_label_buffer = rftk.buffers.as_vector_buffer(train_pixel_labels)

        for i, number_of_trees in enumerate(number_of_trees_list):
            if number_of_features_list is not None:
                predictor = learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      classes=train_pixel_label_buffer,
                                      bootstrap=bootstrap,
                                      number_of_trees=number_of_trees, 
                                      number_of_features=number_of_features_list[i], 
                                      number_of_jobs=number_of_jobs)
            else:
                predictor = learner.fit(depth_images=train_depths_buffer, 
                                      pixel_indices=train_pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      classes=train_pixel_label_buffer,
                                      bootstrap=bootstrap,
                                      number_of_trees=number_of_trees, 
                                      number_of_jobs=number_of_jobs)

        train_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
        train_accurracy = np.mean(train_pixel_labels == train_predictions.argmax(axis=1))
        test_predictions = predictor.predict(depth_images=rftk.buffers.as_tensor_buffer(test_depths),
                                        pixel_indices=rftk.buffers.as_matrix_buffer(test_pixel_indices))
        test_accurracy = np.mean(test_pixel_labels == test_predictions.argmax(axis=1))

        if bootstrap:

            train_predictions_oob = predictor.predict_oob(depth_images=rftk.buffers.as_tensor_buffer(train_depths),
                                            pixel_indices=rftk.buffers.as_matrix_buffer(train_pixel_indices))
            train_accurracy_oob = np.mean(train_pixel_labels == train_predictions_oob.argmax(axis=1))

            print("%s %f %f %f (#trees=%d)" % (description, train_accurracy, train_accurracy_oob, test_accurracy, predictor.get_forest().GetNumberOfTrees()))
        else:
            print("%s %f %f (#trees=%d)" % (description, train_accurracy, test_accurracy, predictor.get_forest().GetNumberOfTrees()))

        return test_accurracy

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

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                        extractor_type='axis_aligned',
                                                                        prediction_type='classification',
                                                                        split_type='all_midpoints',
                                                                        tree_type='depth_first'),
                        description="ecoli create_uber_learner axis_aligned",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_vanilia_classifier(),
                        description="ecoli create_vanilia_classifier",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)

        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                        extractor_type='axis_aligned',
                                                                        prediction_type='classification',
                                                                        split_type='all_midpoints',
                                                                        tree_type='depth_first'),
                        description="ecoli create_uber_learner axis_aligned bootstrap",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_vanilia_classifier(),
                        description="ecoli create_vanilia_classifier bootstrap",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                        extractor_type='axis_aligned',
                                                                        prediction_type='classification',
                                                                        split_type='all_midpoints',
                                                                        tree_type='breadth_first'),
                        description="ecoli create_uber_learner axis_aligned bootstrap breadth_first",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_vanilia_classifier(tree_order='breadth_first'),
                        description="ecoli create_vanilia_classifier bootstrap tree_order==breadth_first",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                        extractor_type='dimension_pair_diff',
                                                                        prediction_type='classification',
                                                                        split_type='all_midpoints',
                                                                        tree_type='depth_first'),
                        description="ecoli create_uber_learner dimension_pair_diff",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)


        error = run_classifier(learner=rftk.learn.create_dimension_pair_difference_matrix_classifier(),
                        description="ecoli create_dimension_pair_difference_matrix_classifier",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)

        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                extractor_type='dimension_pair_diff',
                                                                prediction_type='classification',
                                                                split_type='all_midpoints',
                                                                tree_type='depth_first'),
                description="ecoli create_uber_learner dimension_pair_diff bootstrap",
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_dimension_pair_difference_matrix_classifier(),
                        description="ecoli create_dimension_pair_difference_matrix_classifier bootstrap",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                extractor_type='class_pair_diff',
                                                                prediction_type='classification',
                                                                split_type='all_midpoints',
                                                                tree_type='depth_first'),
                description="ecoli create_uber_learner class_pair_diff bootstrap",
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_class_pair_difference_matrix_classifier(),
                        description="ecoli create_class_pair_difference_matrix_classifier bootstrap",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=True)
        self.assertGreater(error, 0.75)

        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                extractor_type='axis_aligned',
                                                                prediction_type='classification',
                                                                split_type='constant_splitpoints',
                                                                constant_splitpoints_type='at_random_datapoints',
                                                                number_of_splitpoints=10,
                                                                streams_type='one_stream',
                                                                tree_type='depth_first'),
                description="ecoli create_uber_learner axis_aligned constant_splitpoints at_random_datapoints one_stream",
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_one_stream_classifier(number_of_splitpoints=10),
                        description="ecoli create_one_stream_classifier",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)

        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                extractor_type='axis_aligned',
                                                                prediction_type='classification',
                                                                split_type='constant_splitpoints',
                                                                constant_splitpoints_type='at_random_datapoints',
                                                                number_of_splitpoints=10,
                                                                streams_type='two_stream_per_tree',
                                                                tree_type='depth_first'),
                description="ecoli create_uber_learner axis_aligned constant_splitpoints at_random_datapoints two_stream_per_tree",
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_two_stream_classifier(),
                        description="ecoli create_two_stream_classifier",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], bootstrap=False)
        self.assertGreater(error, 0.75)

        #######################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                extractor_type='axis_aligned',
                                                                prediction_type='classification',
                                                                split_type='constant_splitpoints',
                                                                constant_splitpoints_type='at_random_datapoints',
                                                                number_of_splitpoints=10,
                                                                streams_type='one_stream',
                                                                tree_type='online'),
                description="ecoli create_uber_learner axis_aligned constant_splitpoints at_random_datapoints one_stream online",
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                number_of_trees_list=[200,200,200], bootstrap=False)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_online_one_stream_classifier(
                                                number_of_splitpoints=100,
                                                min_impurity=0.1, 
                                                min_child_size_sum=3,
                                                max_frontier_size=50000 ),
                        description="ecoli create_online_one_stream_classifier",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200,200,200], bootstrap=True)
        self.assertGreater(error, 0.7)


        ########################################################

        error = run_classifier(learner=rftk.learn.create_uber_learner(  data_type='matrix', 
                                                                extractor_type='axis_aligned',
                                                                prediction_type='classification',
                                                                possion_number_of_features=True,
                                                                split_type='constant_splitpoints',
                                                                constant_splitpoints_type='at_random_datapoints',
                                                                number_of_splitpoints=10,
                                                                streams_type='two_stream_per_tree',
                                                                tree_type='online'),
                description="ecoli create_uber_learner axis_aligned constant_splitpoints at_random_datapoints two_stream_per_tree online",
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                number_of_trees_list=[200,200,200], bootstrap=False)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_online_two_stream_consistent_classifier(
                                                    number_of_splitpoints=100,
                                                    min_impurity=0.1, 
                                                    poisson_sample=1.0, 
                                                    number_of_data_to_split_root=3, 
                                                    number_of_data_to_force_split_root=20, 
                                                    split_rate_growth=1.001,
                                                    probability_of_impurity_stream=0.5,
                                                    max_frontier_size=50000 ),
                        description="ecoli create_online_two_stream_consistent_classifier",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200,200,200], bootstrap=True)
        self.assertGreater(error, 0.7)


    def test_ecoli_greedy_classifiers(self):
        x_train, y_train, x_test, y_test = load_data.load_ecoli_data()

        error = run_classifier(learner=rftk.learn.create_greedy_add_swap_classifier(),
                        description="ecoli create_greedy_add_swap_classifier bootstrap",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[1,49,50,50,50], bootstrap=True)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_fast_greedy_add_swap_classifier(),
                        description="ecoli create_fast_greedy_add_swap_classifier bootstrap",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[1,49,50,50,50], bootstrap=True)
        self.assertGreater(error, 0.75)


        error = run_classifier(learner=rftk.learn.create_fast_greedy_add_swap_classifier(),
                        description="ecoli create_fast_greedy_add_swap_classifier bootstrap diff feature values",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[1,49,50,50,50], 
                        number_of_features_list=[2,3,1,3,2],
                        bootstrap=True)
        self.assertGreater(error, 0.75)


        error = run_classifier(learner=rftk.learn.create_vanilia_classifier(),
                        description="ecoli create_vanilia_classifier bootstrap",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[200], 
                        bootstrap=True)
        self.assertGreater(error, 0.75)

        error = run_classifier(learner=rftk.learn.create_vanilia_classifier(),
                        description="ecoli create_vanilia_classifier single tree",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        number_of_trees_list=[1],
                        number_of_features_list=[x_train.shape[1]],
                        bootstrap=True)
        self.assertGreater(error, 0.70)




    def test_usps_greedy_classifiers(self):
        x_train, y_train, x_test, y_test = load_data.load_usps_data()

        # error = run_classifier(learner=rftk.learn.create_greedy_add_swap_classifier(),
        #                         description="usps create_greedy_add_swap_classifier bootstrap",
        #                         x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        #                         number_of_trees_list=[1,9,10,10,10], bootstrap=True)
        # self.assertGreater(error, 0.91)

        print "Current date and time: " , datetime.datetime.now()

        error = run_classifier(learner=rftk.learn.create_fast_greedy_add_swap_classifier(),
                                description="usps create_fast_greedy_add_swap_classifier bootstrap",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[1,99], bootstrap=True)
        self.assertGreater(error, 0.91)

        print "Current date and time: " , datetime.datetime.now()

        error = run_classifier(learner=rftk.learn.create_vanilia_classifier(),
                                description="usps create_vanilia_classifier bootstrap",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[100], bootstrap=True)

        print "Current date and time: " , datetime.datetime.now()

        self.assertGreater(error, 0.91)
        
        # error = run_classifier(learner=rftk.learn.create_vanilia_classifier(),
        #                         description="usps create_vanilia_classifier single tree",
        #                         x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        #                         number_of_trees_list=[1], number_of_features_list=[x_train.shape[1]])
        # self.assertGreater(error, 0.85)


    def test_wine_regression(self):
        x_train, y_train, x_test, y_test = load_data.load_wine_data()

        error = run_regression(learner=rftk.learn.create_standard_regression(),
                                description="wine create_standard_regression",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[20], bootstrap=False)
        self.assertLess(error, 0.5)

        error = run_regression(learner=rftk.learn.create_standard_regression(),
                                description="wine create_standard_regression bootstrap",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[100], bootstrap=True)
        self.assertLess(error, 0.5)

        error = run_regression(learner=rftk.learn.create_biau2008_regression(),
                                description="wine create_biau2008_regression",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[20], bootstrap=False)
        self.assertLess(error, 0.5)

        error = run_regression(learner=rftk.learn.create_consistent_regression(),
                                description="wine create_consistent_regression",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[20], bootstrap=False)
        self.assertLess(error, 0.5)

        error = run_regression(learner=rftk.learn.create_consistent_regression(),
                                description="wine create_consistent_regression",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[20], bootstrap=False)
        self.assertLess(error, 0.5)



        x_train, y_train, x_test, y_test = load_data.load_wine_data(normalize_data=True)
        error = run_regression(learner=rftk.learn.create_biau2012_regression(number_of_jobs=1, min_impurity=-1, min_node_size=-1),
                                description="wine create_biau2012_regression",
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                number_of_trees_list=[20], bootstrap=False)
        self.assertLess(error, 0.8)





    def test_vanilia_scaled_depth_delta_classifier(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        

        forest_learner = rftk.learn.create_vanilia_scaled_depth_delta_classifier(
                                      number_of_trees=5,
                                      number_of_features=1000,
                                      min_impurity_gain=0.1,
                                      # max_depth=30,
                                      min_samples_split=5,
                                      # min_samples_leaf = 5,
                                      # min_impurity_gain = 0.01
                                      ux=75, uy=75, vx=75, vy=75,
                                      bootstrap=True,
                                      number_of_jobs=5)

        error = run_depth_delta_classifier(forest_learner, "create_vanilia_scaled_depth_delta_classifier 2", 
                            train_depths=train_depths, train_labels=train_labels, train_pixel_indices=train_pixel_indices, train_pixel_labels=train_pixel_labels, train_joint_offsets=train_joint_offsets,
                            test_depths=test_depths, test_labels=test_labels, test_pixel_indices=test_pixel_indices, test_pixel_labels=test_pixel_labels, test_joint_offsets=test_joint_offsets, 
                            number_of_trees_list=[10], 
                            bootstrap=True,
                            number_of_features_list=[1000],
                            number_of_jobs=5)
        self.assertGreater(error, 0.4)


        forest_learner = rftk.learn.create_uber_learner(    data_type='depth_image', 
                                                            extractor_type='pixel_pair_diff',
                                                            prediction_type='classification',
                                                            split_type='all_midpoints',
                                                            tree_type='depth_first',
                                                            min_impurity_gain=0.1,
                                                            min_samples_split=5,
                                                            ux=75, uy=75, vx=75, vy=75, )


        error = run_depth_delta_classifier(forest_learner, "create_uber_learner depth_image classification all_midpoints", 
                            train_depths=train_depths, train_labels=train_labels, train_pixel_indices=train_pixel_indices, train_pixel_labels=train_pixel_labels, train_joint_offsets=train_joint_offsets,
                            test_depths=test_depths, test_labels=test_labels, test_pixel_indices=test_pixel_indices, test_pixel_labels=test_pixel_labels, test_joint_offsets=test_joint_offsets, 
                            number_of_trees_list=[10], 
                            bootstrap=True,
                            number_of_features_list=[1000],
                            number_of_jobs=5)
        self.assertGreater(error, 0.4)



    def test_online_one_stream_depth_delta_classifier(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()

        forest_learner = rftk.learn.create_online_one_stream_depth_delta_classifier(
                                      number_of_features=1000,
                                      min_impurity_gain=0.1,
                                      # max_depth=30,
                                      min_samples_split=5,
                                      # min_samples_leaf = 5,
                                      # min_impurity_gain = 0.01
                                      ux=75, uy=75, vx=75, vy=75,
                                      bootstrap=True,
                                      number_of_jobs=5)

        error = run_depth_delta_classifier(forest_learner, "create_online_one_stream_depth_delta_classifier 2", 
                            train_depths=train_depths, train_labels=train_labels, train_pixel_indices=train_pixel_indices, train_pixel_labels=train_pixel_labels, train_joint_offsets=train_joint_offsets,
                            test_depths=test_depths, test_labels=test_labels, test_pixel_indices=test_pixel_indices, test_pixel_labels=test_pixel_labels, test_joint_offsets=test_joint_offsets, 
                            number_of_trees_list=[5], 
                            bootstrap=True,
                            number_of_features_list=[1000],
                            number_of_jobs=5)
        self.assertGreater(error, 0.32)


        forest_learner = rftk.learn.create_uber_learner(    data_type='depth_image', 
                                                            extractor_type='pixel_pair_diff',
                                                            prediction_type='classification',
                                                            split_type='constant_splitpoints',
                                                            constant_splitpoints_type='at_random_datapoints',
                                                            number_of_splitpoints=10,
                                                            streams_type='one_stream',
                                                            tree_type='online',
                                                            min_impurity_gain=0.1,
                                                            min_samples_split=5,
                                                            ux=75, uy=75, vx=75, vy=75, )

        error = run_depth_delta_classifier(forest_learner, "create_uber_learner depth_image classification at_random_datapoints one_stream", 
                            train_depths=train_depths, train_labels=train_labels, train_pixel_indices=train_pixel_indices, train_pixel_labels=train_pixel_labels, train_joint_offsets=train_joint_offsets,
                            test_depths=test_depths, test_labels=test_labels, test_pixel_indices=test_pixel_indices, test_pixel_labels=test_pixel_labels, test_joint_offsets=test_joint_offsets, 
                            number_of_trees_list=[5], 
                            bootstrap=True,
                            number_of_features_list=[1000],
                            number_of_jobs=5)
        self.assertGreater(error, 0.32)


    def test_online_two_stream_consistent_depth_delta_classifier(self):
        train_depths, train_labels, train_pixel_indices, train_pixel_labels, train_joint_offsets = load_data.load_kinect_train_data()
        test_depths, test_labels, test_pixel_indices, test_pixel_labels, test_joint_offsets = load_data.load_kinect_test_data()
        
        forest_learner = rftk.learn.create_online_two_stream_consistent_depth_delta_classifier(
                                      number_of_features=1000,
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

        error = run_depth_delta_classifier(forest_learner, "create_online_two_stream_consistent_depth_delta_classifier 2", 
                            train_depths=train_depths, train_labels=train_labels, train_pixel_indices=train_pixel_indices, train_pixel_labels=train_pixel_labels, train_joint_offsets=train_joint_offsets,
                            test_depths=test_depths, test_labels=test_labels, test_pixel_indices=test_pixel_indices, test_pixel_labels=test_pixel_labels, test_joint_offsets=test_joint_offsets, 
                            number_of_trees_list=[5], 
                            bootstrap=True,
                            number_of_features_list=[1000],
                            number_of_jobs=5)
        self.assertGreater(error, 0.32)


        forest_learner = rftk.learn.create_uber_learner(    data_type='depth_image', 
                                                            extractor_type='pixel_pair_diff',
                                                            prediction_type='classification',
                                                            possion_number_of_features=True,
                                                            split_type='constant_splitpoints',
                                                            constant_splitpoints_type='at_random_datapoints',
                                                            number_of_splitpoints=10,
                                                            streams_type='two_stream_per_tree',
                                                            tree_type='online',
                                                            min_impurity_gain=0.1,
                                                            min_samples_split=5,
                                                            ux=75, uy=75, vx=75, vy=75, )

        error = run_depth_delta_classifier(forest_learner, "create_uber_learner depth_image classification at_random_datapoints two_stream_tree", 
                            train_depths=train_depths, train_labels=train_labels, train_pixel_indices=train_pixel_indices, train_pixel_labels=train_pixel_labels, train_joint_offsets=train_joint_offsets,
                            test_depths=test_depths, test_labels=test_labels, test_pixel_indices=test_pixel_indices, test_pixel_labels=test_pixel_labels, test_joint_offsets=test_joint_offsets, 
                            number_of_trees_list=[5], 
                            bootstrap=True,
                            number_of_features_list=[1000],
                            number_of_jobs=5)
        self.assertGreater(error, 0.32)

      


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








if __name__ == '__main__':
    unittest.main()
