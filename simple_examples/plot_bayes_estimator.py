import argparse

import plot_utils
import dist_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Synthetic')
    parser.add_argument('-d', '--distribution', default="3-class", help='distribution name')
    args = parser.parse_args()

    dist = dist_utils.get_mog_dist(args.distribution)
    n_per = 1000

    X_train,Y_train = dist.sample(n_per)
    X_test,Y_test = dist.sample(n_per)

    plot_utils.grid_plot(dist, X_train, Y_train, X_test, "bayes_estimator_%s.png" % (args.distribution))
    plot_utils.grid_plot(dist, X_train, Y_train, X_test, "bayes_estimator_%s-scatter.png" % (args.distribution), plot_scatter=True)

