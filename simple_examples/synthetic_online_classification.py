import numpy as np
import argparse
import os
import cPickle as pickle
from datetime import datetime
import random as random

import sklearn.datasets as sklearn_datasets

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data  as forest_data
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train


import rftk.utils.predict as predict_utils
import rftk.utils.forest as forest_utils

import dist_utils

def load_data(file):
    data = sklearn_datasets.load_svmlight_file(file)
    X = np.array(data[0].todense(),  dtype=np.float32)
    Y = np.array( data[1], dtype=np.int32 )
    (m,n) = X.shape
    return X, Y, m, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Synthetic')
    parser.add_argument('-d', '--distribution', default="3-class", help='distribution name')
    parser.add_argument('-n', '--number_of_trees', default=25, type=int, help='number of epochs')
    parser.add_argument('-m', '--max_epoch', default=18, type=int, help='number of epochs')
    parser.add_argument('-r', '--split_rate', default=1.2, type=float, help='amount of extra data required at depth+1')
    parser.add_argument('-s', '--amount_of_data_to_split_first', default=2.0, type=float, help='amount of data required for first split')
    parser.add_argument('-i', '--independent_impurity_and_ys', default=1, type=int, help='whether to use 2 streams')
    parser.add_argument('-pn', '--null_probability', default=0.1, type=float, help='probability that each tree ignores datapoint')
    parser.add_argument('-pi', '--impurity_probability', default=0.5, type=float, help='probability of impurity stream')
    parser.add_argument('-t', '--number_of_thresholds', default=10, type=int, help='number of thresholds')
    args = parser.parse_args()

    unique_name = ("%s-%d-%d-%.2f-%.2f-%d-%.2f-%.2f-%d-%s") % (
                      args.distribution, args.number_of_trees, args.max_epoch,
                      args.split_rate, args.amount_of_data_to_split_first,
                      args.independent_impurity_and_ys, args.null_probability, args.impurity_probability,
                      args.number_of_thresholds,
                      str(datetime.now()).replace(':', '-').replace(' ', '-'))

    data_path = ("online-%s-data") % (unique_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Sample data
    # dist = dist_utils.get_mog_dist(args.distribution)
    # train_size = 2**(args.max_epoch)
    # test_size = 100000
    X_train, Y_train, number_of_samples, number_of_attributes = load_data("usps")

    row_indices = range(number_of_samples)
    random.shuffle( row_indices )
    X_train = X_train[row_indices, :]
    Y_train = Y_train[row_indices]

    print X_train.shape
    random.shuffle( row_indices )
    X_train = np.append(X_train, X_train[row_indices, :], axis=0)
    Y_train = np.append(Y_train, Y_train[row_indices])
    random.shuffle( row_indices )
    X_train = np.append(X_train, X_train[row_indices, :], axis=0)
    Y_train = np.append(Y_train, Y_train[row_indices])

    X_test, Y_test,_,_= load_data("usps.t")
    print "HEREH"
    print X_train.shape
    print X_test.shape
    # Save data
    # print datetime.now()
    # X_train,Y_train = dist.sample(train_size)
    # print datetime.now()
    # X_test,Y_test = dist.sample(test_size)


    print datetime.now()
    pickle.dump((X_train, Y_train, X_test, Y_test), open("%s/data.pkl" % (data_path), "wb"))
    print datetime.now()

    # Bayes accuracy
    print "Computing bayes accuracy"
    # y_probs = dist.predict_proba(X_test)
    # y_hat = y_probs.argmax(axis=1)
    # bayes_accuracy = np.mean(Y_test == y_hat)
    bayes_accuracy = 1
    print "Bayes accuracy %0.5f" % bayes_accuracy
    forest_accuracy = np.zeros(args.max_epoch)
    tree_accuracy = np.zeros((args.max_epoch, args.number_of_trees))

    # Configure online forest
    max_features=int(np.sqrt(number_of_attributes))
    max_depth=15
    min_impurity=0.001
    # min_samples_split=10
    (_,x_dim) = X_test.shape
    # y_dim = dist.number_classes
    y_dim = int(np.max(Y_train) + 1)
    number_of_jobs = 1
    # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( max_features, x_dim, x_dim, True)
    feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( max_features, x_dim, True)
    # node_data_collector = train.AllNodeDataCollectorFactory()
    # class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)
    if args.independent_impurity_and_ys == 0:
        node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                args.number_of_thresholds,
                                                                                args.null_probability)
        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT, buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT)
    else:
        node_data_collector = train.TwoStreamRandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                            args.number_of_thresholds,
                                                                                            args.null_probability,
                                                                                            args.impurity_probability)
        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                buffers.IMPURITY_HISTOGRAM_LEFT, buffers.IMPURITY_HISTOGRAM_RIGHT,
                buffers.YS_HISTOGRAM_LEFT, buffers.YS_HISTOGRAM_RIGHT)

    # self.split_criteria = train.OnlineAlphaBetaSplitCriteria(   max_depth,
    #                                                             min_impurity,
    #                                                             min_samples_split)
    split_criteria = train.OnlineConsistentSplitCriteria(  args.split_rate,
                                                                min_impurity,
                                                                args.amount_of_data_to_split_first,
                                                                10 * args.amount_of_data_to_split_first)

    extractor_list = [feature_extractor]
    train_config = train.TrainConfigParams(extractor_list,
                                            node_data_collector,
                                            class_infogain_best_split,
                                            split_criteria,
                                            args.number_of_trees,
                                            100000)
    sampling_config = train.OnlineSamplingParams(False, 1.0)
    online_learner = train.OnlineForestLearner(train_config)

    total_samples = 0
    total_sample_list = []
    for epoch_id in range(1,args.max_epoch):
        print "Fitting epoch %d" % (epoch_id)

        # Construct data for update
        epoch_per = min(2**epoch_id, number_of_samples*3) - total_samples
        if epoch_per > 0:

          total_samples += epoch_per
          total_sample_list.append(total_samples)
          X_online_train_sample = X_train[total_samples-epoch_per:total_samples, :]
          Y_online_train_sample = Y_train[total_samples-epoch_per:total_samples]
          (x_m,_) = X_online_train_sample.shape

          # Train online forest
          data = buffers.BufferCollection()
          data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffers.as_matrix_buffer(X_online_train_sample))
          data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.Int32Vector(Y_online_train_sample))
          indices = buffers.Int32Vector( np.array(np.arange(x_m), dtype=np.int32) )
          online_learner.Train(data, indices, sampling_config)

          # pickle online forest for later evaluation
          forest_pickle_filename = "%s/forest-%d.pkl" % (data_path, total_samples)
          forest_utils.pickle_dump_native_forest(online_learner.GetForest(), forest_pickle_filename)

          print datetime.now()

          #
          print "Synthetic (classification):"
          predict_forest = predict_utils.MatrixForestPredictor(online_learner.GetForest())
          y_probs = predict_forest.predict_proba(X_test)
          y_hat = y_probs.argmax(axis=1)
          print "    Accuracy:", np.mean(Y_test == y_hat)
          forest_accuracy[epoch_id] = np.mean(Y_test == y_hat)

          print datetime.now()
          print "Predict trees"
          online_forest_data = online_learner.GetForest()
          for tree_id in range(online_forest_data.GetNumberOfTrees()):
              single_tree_forest_data = forest_data.Forest([online_forest_data.GetTree(tree_id)])
              single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
              y_probs = single_tree_forest_predictor.predict_proba(X_test)
              y_hat = y_probs.argmax(axis=1)
              tree_accuracy[epoch_id,tree_id] = np.mean(Y_test == y_hat)

          print datetime.now()

          # pickle accuracies for plotting
          print "pickle trees"
          pickle.dump((epoch_id, total_sample_list, forest_accuracy, tree_accuracy, bayes_accuracy), open("%s/accuracies.pkl" % (data_path), "wb"))


