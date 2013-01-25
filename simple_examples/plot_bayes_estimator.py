import plot_utils
import dist_utils

if __name__ == "__main__":
    dist = dist_utils.mog_2d_3class_example1()
    n_per = 1000

    X_train,Y_train = dist.sample(n_per)
    X_test,Y_test = dist.sample(n_per)

    plot_utils.grid_plot(dist, X_train, Y_train, X_test, "bayes_estimator_mog_2d_3class_example1.png")

