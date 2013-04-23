import numpy as np
import argparse
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import experiment_utils
import plot_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare tree performance to the forest.')
    parser.add_argument('-f', '--forest',
        help="forest file",
        required=True)
    parser.add_argument('-t', '--trees',
        help="tree file",
        required=True)
    parser.add_argument('--out',
        help="figure file name",
        required=True)
    parser.add_argument('-q',
        help="suppress interactive display",
        action="store_true",
        required=False)
    args = parser.parse_args()

    print "Loading data..."
    with open(args.forest) as forest_file:
        forest = pickle.load(forest_file)
    forest_measurement_grid = forest['measurements']

    with open(args.trees) as trees_file:
        trees = pickle.load(trees_file)
    tree_measurement_grids = [x['measurements'] for x in trees]


    print "Combining tree measurement grids..."
    combined_tree_measurement_grid = \
        experiment_utils.management.join_grids_along_new_axis(
            "tree_id", tree_measurement_grids)

    print "Plotting..."
    params = {
            'axes.labelsize' : 20,
            'font.size' : 20,
            'text.fontsize' : 20,
            'legend.fontsize': 20,
            'xtick.labelsize' : 15,
            'ytick.labelsize' : 15,
            }
    matplotlib.rcParams.update(params)


    fig = plt.figure()
    ax = fig.gca()

    plot_utils.draw_line_with_uncertainty(
        ax,
        combined_tree_measurement_grid,
        x='data_size',
        y='accuracy',
        reduce=['tree_id', 'job_id'],
        fill_color='r',
        line_style='r--D',
        label="Trees"
        )

    plot_utils.draw_line_with_uncertainty(
        ax,
        forest_measurement_grid,
        x='data_size',
        y='accuracy',
        reduce=['job_id'],
        fill_color='b',
        line_style='-^',
        label="Forest"
        )

    # plot a "bayes" line
    x = [
        np.min(forest_measurement_grid.domain.extent['data_size']),
        np.max(forest_measurement_grid.domain.extent['data_size'])
        ]
    y = [forest['run_config'].bayes_accuracy]*2
    ax.plot(x, y, linewidth=4, label="Bayes")
    

    ax.set_xlabel('Data Size')
    ax.set_ylabel('Accuracy')

    x_extent = forest['measurements'].domain.extent['data_size']
    ax.set_xlim([x_extent.min(), x_extent.max()])
    ax.set_xscale('log')
    ax.legend(loc="lower right")
    ax.set_title("Forest and tree accuracy")

    plt.savefig(args.out)
    if not args.q:
        plt.show()

