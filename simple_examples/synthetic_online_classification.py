import numpy as np
import matplotlib.pyplot as plt
import random

from datetime import datetime

import rftk.native.forest_data as forest_data
import rftk.native.predict as predict
import rftk.utils.predict as predict_utils
import rftk.utils.sklearnimposteronline as rf
import utils


def build_data(n_per):
    X_1 = np.random.standard_normal(size=(n_per, 2)) + np.array([3, 1])
    X_2 = np.random.standard_normal(size=(n_per, 2)) + np.array([1.5, -2])
    X_3 = np.random.standard_normal(size=(n_per, 2)) + np.array([-1, 0.5])
    Y_1 = np.zeros(n_per)
    Y_2 = np.ones(n_per)
    Y_3 = np.ones(n_per) * 2
    X = np.concatenate([X_1, X_2, X_3], axis=0)
    Y = np.concatenate([Y_1, Y_2, Y_3], axis=0)
    Xfloat32 = np.array( X, dtype=np.float32)
    y = np.array( Y, dtype=np.int32 )

    row_indices = range(n_per*3)
    random.shuffle( row_indices )
    shuffled_Xfloat32 = Xfloat32[row_indices, :]
    shuffled_y = y[row_indices]
    return shuffled_Xfloat32, shuffled_y


def grid_plot(predict_forest, X_train, Y_train, X_test, plot_filename):
    grid_extend = [X_test[:,0].min(), X_test[:,0].max(), X_test[:,1].min(), X_test[:,1].max()]
    Ux, Uy = np.meshgrid(
            np.linspace(grid_extend[0], grid_extend[1]),
            np.linspace(grid_extend[2], grid_extend[3]),
        )

    X_grid = np.concatenate([
        Ux.reshape((-1,1)), Uy.reshape((-1,1))],
        axis=1)

    Y_probs = predict_utils.vec_predict_ys(predict_forest, X_grid)
    Y_hat = utils.max_of_n_prediction(Y_probs)

    print datetime.now()

    plt.figure()
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    img = utils.image_from_predictions(Y_hat, utils.max_prob_of_n_prediction(Y_probs), colors, Ux.shape)
    plt.imshow(img, extent=grid_extend, origin='lower')
    plt.scatter(X_train[:,0], X_train[:,1], c=utils.colors_from_predictions(Y_train, colors))
    plt.savefig(plot_filename)


if __name__ == "__main__":
    n_per = 1000

    X_test,Y_test = build_data(n_per)

    print datetime.now()

    forest = rf.OnlineRandomForestClassifier(max_features=1, n_estimators=100, max_depth=15, min_impurity=0.001, min_samples_split=10, x_dim=2, y_dim=3)
    for epoch_id in range(1,1000):
        print "Fitting epoch %d" % (epoch_id)
        epoch_per = 10 * epoch_id
        X_train,Y_train = build_data(epoch_per)
        forest.fit(X_train, Y_train)

        print datetime.now()

        y_probs = forest.predict(X_test)
        y_hat = utils.max_of_n_prediction(y_probs)

        print "Synthetic (classification):"
        print "    Accuracy:", np.mean(Y_test == y_hat)

        grid_plot(forest.predict_forest, X_train, Y_train, X_test, "output-all/synthetic_online_classification-%d.png" % (epoch_per))

        online_forest_data = forest.online_learner.GetForest()

        for tree_id in range(online_forest_data.GetNumberOfTrees()):
            print tree_id
            single_tree_forest = forest_data.Forest([online_forest_data.GetTree(tree_id)])
            single_tree_forest_predictor = predict.ForestPredictor(single_tree_forest)
            grid_plot(single_tree_forest_predictor, X_train, Y_train, X_test, "output-trees/synthetic_online_classification-%d-%d.png" % (epoch_per, tree_id))