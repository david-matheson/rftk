import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import rftk.utils.sklearnimposter as sklearnimposter
import utils

if __name__ == "__main__":
    n_per = 2000

    X_1 = np.random.standard_normal(size=(n_per, 2)) + np.array([2, 1])
    X_2 = np.random.standard_normal(size=(n_per, 2)) + np.array([0.5, -2])
    X_3 = np.random.standard_normal(size=(n_per, 2)) + np.array([-2, 0.5])
    Y_1 = np.zeros(n_per)
    Y_2 = np.ones(n_per)
    Y_3 = np.ones(n_per) * 2
    X = np.concatenate([X_1, X_2, X_3], axis=0)
    Y = np.concatenate([Y_1, Y_2, Y_3], axis=0)
    Xfloat32 = np.array( X, dtype=np.float32)
    y = np.array( Y, dtype=np.int32 )

    print datetime.now()

    forest = sklearnimposter.RandomForestClassifier(max_features=1, n_estimators=50, max_depth=15, min_samples_split=5)
    forest.fit(Xfloat32, y)
    print datetime.now()
    y_probs = forest.predict(Xfloat32)
    y_hat = utils.max_of_n_prediction(y_probs)

    print "Synthetic (classification):"
    print "    Accuracy:", np.mean(y == y_hat)

    grid_extend = [X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()]
    Ux, Uy = np.meshgrid(
            np.linspace(grid_extend[0], grid_extend[1]),
            np.linspace(grid_extend[2], grid_extend[3]),
        )

    X_grid = np.concatenate([
        Ux.reshape((-1,1)), Uy.reshape((-1,1))],
        axis=1)

    Y_probs = forest.predict(X_grid)
    Y_hat = utils.max_of_n_prediction(Y_probs)

    print datetime.now()

    plt.figure()
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    img = utils.image_from_predictions(Y_hat, utils.max_prob_of_n_prediction(Y_probs), colors, Ux.shape)
    plt.imshow(img, extent=grid_extend, origin='lower')
    plt.scatter(X[:,0], X[:,1], c=utils.colors_from_predictions(y, colors))
    plt.savefig("synthetic_classification.png")
