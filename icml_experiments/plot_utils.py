import numpy as np
import matplotlib.pyplot as plt
import itertools

def colors_from_predictions(Y_hat, colors):
    """
    Y_hat: A length n numpy array where Y_hat[i] is the class with the highest
    probability
    colors: List of rgb values to assign for each class

    Returns: A size (n_data,3) numpy vector y_colors where y_colors[i] is the color
    of the class Y_hat[i]
    """
    y_colors = np.zeros((len(Y_hat),3))
    for yi in range(len(colors)):
        y_colors[ Y_hat == yi ] = colors[yi]
    return y_colors


def image_from_predictions(Y_hat, Y_probs, colors, shape):
    """
    Y_hat: A length n numpy array where Y_hat[i] is the class with the highest
    probability
    Y_probs:  A length n numpy array where Y_probs[i] is the probability of Y_hat[i]
    colors: List of rgb values to assign for each class
    shape: 2 dimension shape of output image

    Returns: A size (shape[0],shape[1],3) numpy vector img where img[i,j] is
    the color of the class Y_hat[i*shape[1] + j] and shaded by the probability
    """
    img_flat = colors_from_predictions(Y_hat, colors)
    # darken colors by probability
    img_flat = (img_flat.T * Y_probs.T * Y_probs.T).T

    img = img_flat.reshape((shape[0],shape[1],3))
    return img


def grid_plot(predictor, X_train, Y_train, X_test, plot_filename, plot_scatter=False):
    grid_extend = [X_test[:,0].min(), X_test[:,0].max(), X_test[:,1].min(), X_test[:,1].max()]
    Ux, Uy = np.meshgrid(
            np.linspace(grid_extend[0], grid_extend[1]),
            np.linspace(grid_extend[2], grid_extend[3]),
        )

    X_grid = np.concatenate([
        Ux.reshape((-1,1)), Uy.reshape((-1,1))],
        axis=1)

    Y_probs = predictor.predict_proba(X_grid)
    Y_hat = Y_probs.argmax(axis=1)

    plt.figure()
    colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1]])
    img = image_from_predictions(Y_hat, Y_probs.max(axis=1), colors, Ux.shape)
    plt.imshow(img, extent=grid_extend, origin='lower')
    if plot_scatter:
        plt.scatter(X_train[:,0], X_train[:,1], c=colors_from_predictions(Y_train, colors))
    plt.savefig(plot_filename)
    plt.close()


def plot_forest_and_tree_accuracy(bayes_accuracy, number_of_datapoints, number_of_passes_list, measurements,
                                plot_standard_deviation, plot_trees, plot_sklearn, plot_offline,
                                log_scale, plot_filename):

    import matplotlib.pyplot as plt
    import experiment_measurement as exm

    if bayes_accuracy is not None:
        bayes_points = np.zeros(len(number_of_datapoints))
        bayes_points.fill(bayes_accuracy)
        plt.plot(number_of_datapoints, bayes_points, '-', lw=2, color='g', label="Bayes estimator")

    for number_of_passes, line_type in zip(number_of_passes_list, ['-', '-.', '.-.', '--']):

        means = np.zeros(len(number_of_datapoints))
        stds = np.zeros(len(number_of_datapoints))

        # plot online random forests
        for i, sample_count in enumerate(number_of_datapoints):
            filered_list = filter(lambda x: isinstance(x, exm.OnlineForestMeasurement)
                and x.number_of_samples == sample_count
                and x.number_of_passes == number_of_passes, measurements)
            values = [x.accuracy for x in filered_list]
            assert(len(values) > 0)
            means[i] = np.mean( values)
            stds[i] = np.std( values )
        plt.plot(number_of_datapoints, means, line_type, lw=2, color='b', label="Online rf %d" % number_of_passes)
        if plot_standard_deviation:
            plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='b')


        # plot online trees
        if plot_trees:
            for i, sample_count in enumerate(number_of_datapoints):
                filered_list = filter(lambda x: isinstance(x, exm.OnlineTreeMeasurement)
                    and x.number_of_samples == sample_count
                    and x.number_of_passes == number_of_passes, measurements)
                values = [x.accuracy for x in filered_list]
                assert(len(values) > 0)
                means[i] = np.mean( values)
                stds[i] = np.std( values )
            plt.plot(number_of_datapoints, means, line_type, lw=1, color='r', label="Online trees %d" % number_of_passes)
            if plot_standard_deviation:
                plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='r')

        if plot_offline:
            for i, sample_count in enumerate(number_of_datapoints):
                filered_list = filter(lambda x: isinstance(x, exm.OfflineForestMeasurement)
                    and x.number_of_samples == sample_count
                    and x.number_of_passes == number_of_passes, measurements)
                values = [x.accuracy for x in filered_list]
                assert(len(values) > 0)
                means[i] = np.mean( values)
                stds[i] = np.std( values )
            plt.plot(number_of_datapoints, means, line_type, lw=1, color='green', label="Offline forest %d" % number_of_passes)
            if plot_standard_deviation:
                plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='green')

        # plot sklearn forests
        if plot_sklearn:
            for i, sample_count in enumerate(number_of_datapoints):
                filered_list = filter(lambda x: isinstance(x, exm.SklearnForestMeasurement)
                    and x.number_of_samples == sample_count
                    and x.number_of_passes == number_of_passes, measurements)
                values = [x.accuracy for x in filered_list]
                assert(len(values) > 0)
                means[i] = np.mean( values)
                stds[i] = np.std( values )
            plt.plot(number_of_datapoints, means, line_type, lw=1, color='grey', label="Sklearn forest %d" % number_of_passes)
            if plot_standard_deviation:
                plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='grey')


    if log_scale:
        plt.xscale('log')

    plt.legend(loc = (0.5, 0.2))
    plt.savefig(plot_filename)
    plt.show()


def plot_depths(number_of_datapoints, number_of_passes_list, measurements, plot_standard_deviation, log_scale, plot_filename):

    import matplotlib.pyplot as plt
    import experiment_measurement as exm

    for number_of_passes, line_type in zip(number_of_passes_list, ['-', '-.', '.-.', '--']):

        means = np.zeros(len(number_of_datapoints))
        stds = np.zeros(len(number_of_datapoints))

        # plot online random forests
        for i, sample_count in enumerate(number_of_datapoints):
            filered_list = filter(lambda x: isinstance(x, exm.OnlineForestStatsMeasurement)
                and x.number_of_samples == sample_count
                and x.number_of_passes == number_of_passes, measurements)
            depths = [x.average_depth for x in filered_list]
            assert(len(depths) > 0)
            means[i] = np.mean( depths )
            stds[i] = np.std( depths )
            print means
        plt.plot(number_of_datapoints, means, line_type, lw=2, color='b', label="orf min depth avg passes=%d" % number_of_passes)
        if plot_standard_deviation:
            plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='b')

        for i, sample_count in enumerate(number_of_datapoints):
            filered_list = filter(lambda x: isinstance(x, exm.OnlineForestStatsMeasurement)
                and x.number_of_samples == sample_count
                and x.number_of_passes == number_of_passes, measurements)
            min_depths = [x.min_depth for x in filered_list]
            assert(len(min_depths) > 0 )
            means[i] = np.mean( min_depths )
            stds[i] = np.std( min_depths )
        plt.plot(number_of_datapoints, means, line_type, lw=2, color='g', label="orf min depth passes=%d" % number_of_passes)
        if plot_standard_deviation:
            plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='g')

        for i, sample_count in enumerate(number_of_datapoints):
            filered_list = filter(lambda x: isinstance(x, exm.OnlineForestStatsMeasurement)
                and x.number_of_samples == sample_count
                and x.number_of_passes == number_of_passes, measurements)
            max_depths = [x.max_depth for x in filered_list]
            assert( len(max_depths) > 0)
            means[i] = np.mean( max_depths )
            stds[i] = np.std( max_depths )
        plt.plot(number_of_datapoints, means, line_type, lw=2, color='r', label="orf max depth passes=%d" % number_of_passes)
        if plot_standard_deviation:
            plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='r')

    if log_scale:
        plt.xscale('log')

    plt.legend(loc = (0.5, 0.0))
    plt.savefig(plot_filename)
    plt.show()


def plot_total_estimator_points(number_of_datapoints, number_of_passes_list, measurements, plot_standard_deviation, log_scale, plot_filename):

    import matplotlib.pyplot as plt
    import experiment_measurement as exm

    for number_of_passes, line_type in zip(number_of_passes_list, ['-', '-.', '.-.', '--']):

        means = np.zeros(len(number_of_datapoints))
        stds = np.zeros(len(number_of_datapoints))

        # plot online random forests
        for i, sample_count in enumerate(number_of_datapoints):
            filered_list = filter(lambda x: isinstance(x, exm.OnlineForestStatsMeasurement)
                and x.number_of_samples == sample_count
                and x.number_of_passes == number_of_passes, measurements)
            total_estimator_points = [x.total_estimator_points for x in filered_list]
            assert(len(total_estimator_points) > 0)
            means[i] = np.mean( total_estimator_points )
            stds[i] = np.std( total_estimator_points )
        plt.plot(number_of_datapoints, means, line_type, lw=2, color='b', label="orf # est points (%d)" % number_of_passes)
        if plot_standard_deviation:
            plt.fill_between(number_of_datapoints, means-stds, means+stds, alpha=0.2, color='b')

    plt.plot(number_of_datapoints, 100*np.array(number_of_datapoints), line_type, lw=2, color='r', label="x=y")


    if log_scale:
        plt.xscale('log')

    plt.legend(loc = (0.5, 0.2))
    plt.savefig(plot_filename)
    plt.show()

