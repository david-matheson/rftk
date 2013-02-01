import numpy as np
import argparse
import os
from datetime import datetime

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data  as forest_data
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train

import rftk.utils.buffer_converters as buffer_converters
import rftk.utils.predict as predict_utils

import plot_utils
import dist_utils

def plot_forest_and_tree(forest_accuracy, tree_accuracy, plot_filename):
    import matplotlib.pyplot as plt

    (_, number_of_trees) = tree_accuracy.shape
    for t in range(number_of_trees):
      print tree_accuracy[:,t]
      plt.plot(np.arange(args.max_epoch), tree_accuracy[:,t], '-', lw=1, color='r')

    print forest_accuracy
    plt.plot(np.arange( args.max_epoch ), forest_accuracy, '-', lw=2, color='b')

    plt.savefig(plot_filename)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Synthetic')
    parser.add_argument('-d', '--distribution', default="3-class", help='distribution name')
    parser.add_argument('-n', '--number_of_trees', default=25, type=int, help='number of epochs')
    parser.add_argument('-m', '--max_epoch', default=20, type=int, help='number of epochs')
    parser.add_argument('-r', '--split_rate', default=1.3, type=float, help='amount of extra data required at depth+1')
    parser.add_argument('-s', '--amount_of_data_to_split_first', default=10.0, type=float, help='amount of data required for first split')
    parser.add_argument('-i', '--independent_impurity_and_ys', default=0, type=int, help='whether to use 2 streams')
    parser.add_argument('-pn', '--null_probability', default=0.5, type=float, help='probability that each tree ignores datapoint')
    parser.add_argument('-pi', '--impurity_probability', default=0.5, type=float, help='probability of impurity stream')
    parser.add_argument('-t', '--number_of_thresholds', default=10, type=int, help='number of thresholds')

    args = parser.parse_args()

    unique_name = ("%s-%d-%d-%.2f-%.2f-%d-%.2f-%.2f-%d-%s") % (
                      args.distribution, args.number_of_trees, args.max_epoch,
                      args.split_rate, args.amount_of_data_to_split_first,
                      args.independent_impurity_and_ys, args.null_probability, args.impurity_probability,
                      args.number_of_thresholds,
                      str(datetime.now()).replace(':', '-').replace(' ', '-'))
    forest_plot_path = ("online-%s-forest") % (unique_name)
    if not os.path.exists(forest_plot_path):
        os.makedirs(forest_plot_path)

    trees_plot_path = ("online-%s-trees") % (unique_name)
    if not os.path.exists(trees_plot_path):
        os.makedirs(trees_plot_path)

    # Sample data
    dist = dist_utils.get_mog_dist(args.distribution)
    train_size = 2**(args.max_epoch)
    test_size = 10000

    print datetime.now()
    X_train,Y_train = dist.sample(train_size)
    print datetime.now()
    X_test,Y_test = dist.sample(test_size)
    print datetime.now()

    # Configure
    max_features=1
    number_of_trees=args.number_of_trees
    max_depth=15
    min_impurity=0.001
    # min_samples_split=10
    (_,x_dim) = X_test.shape
    y_dim = dist.number_classes
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
                                                                (args.split_rate**2) * args.amount_of_data_to_split_first)

    extractor_list = [feature_extractor]
    train_config = train.TrainConfigParams(extractor_list,
                                            node_data_collector,
                                            class_infogain_best_split,
                                            split_criteria,
                                            number_of_trees,
                                            100000)
    sampling_config = train.OnlineSamplingParams(False, 1.0)
    online_learner = train.OnlineForestLearner(train_config)

    forest_accuracy = np.zeros(args.max_epoch)
    tree_accuracy = np.zeros((args.max_epoch, number_of_trees))

    total_samples = 0
    for epoch_id in range(1,args.max_epoch):
        print "Fitting epoch %d" % (epoch_id)
        epoch_per = 2**epoch_id - total_samples
        total_samples += epoch_per
        X_online_train_sample = X_train[total_samples-epoch_per:total_samples, :]
        Y_online_train_sample = Y_train[total_samples-epoch_per:total_samples]
        (x_m,_) = X_online_train_sample.shape

        data = buffers.BufferCollection()
        data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(X_online_train_sample))
        data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.Int32Vector(Y_online_train_sample))
        indices = buffers.Int32Vector( np.array(np.arange(x_m), dtype=np.int32) )
        online_learner.Train(data, indices, sampling_config)
        predict_forest = predict_utils.MatrixForestPredictor(online_learner.GetForest())

        print datetime.now()

        y_probs = predict_forest.predict_proba(X_test)
        y_hat = y_probs.argmax(axis=1)

        print "Synthetic (classification):"
        print "    Accuracy:", np.mean(Y_test == y_hat)
        forest_accuracy[epoch_id] = np.mean(Y_test == y_hat)
        plot_utils.grid_plot(predict_forest, X_online_train_sample, Y_online_train_sample, X_test, "%s/synthetic_online_classification-%d.png" % (forest_plot_path, total_samples))
        plot_utils.grid_plot(predict_forest, X_online_train_sample, Y_online_train_sample, X_test, "%s/synthetic_online_classification-%d-scatter.png" % (forest_plot_path, total_samples),plot_scatter=True)

        online_forest_data = online_learner.GetForest()
        for tree_id in range(online_forest_data.GetNumberOfTrees()):
            print tree_id
            single_tree_forest_data = forest_data.Forest([online_forest_data.GetTree(tree_id)])
            single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
            y_probs = single_tree_forest_predictor.predict_proba(X_test)
            y_hat = y_probs.argmax(axis=1)
            tree_accuracy[epoch_id,tree_id] = np.mean(Y_test == y_hat)
            plot_utils.grid_plot(single_tree_forest_predictor, X_online_train_sample, Y_online_train_sample, X_test, "%s/synthetic_online_classification-%d-%d.png" % (trees_plot_path, total_samples, tree_id))

    plot_forest_and_tree(forest_accuracy, tree_accuracy, "%s/accuracy_over_time.png" % (forest_plot_path))

