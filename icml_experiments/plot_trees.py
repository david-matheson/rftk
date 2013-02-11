import numpy as np
import argparse
import glob
import cPickle as pickle
import matplotlib.pyplot as plt
import experiment_measurement_tensor as experiment_measurement
import plot_utils_tensor as plot_utils


def load_measurements(file_name):
    with open(file_name) as pkl_file:
        data = pickle.load(pkl_file)
    return data['measurements']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare tree performance to the forest.')
    parser.add_argument('-f', '--forest',
        help="forest file",
        required=True)
    parser.add_argument('-t', '--trees',
        help="tree files",
        nargs='+',
        required=True)
    args = parser.parse_args()

    # trees
    print "Loading measurement files..."
    tree_measurement_grids = map(load_measurements, args.trees)

    print "Combining measurement grids..."
    combined_tree_measurement_grid = \
        experiment_measurement.join_grids_along_new_axis("tree_id", tree_measurement_grids)

    print "Plotting..."

    fig = plt.figure()
    ax = fig.gca()

    num_passes = 1

    plot_utils.draw_many_lines_with_uncertainty(
        ax,
        combined_tree_measurement_grid,
        each='tree_id',
        x='data_size',
        y='accuracy',
        slice={'number_of_passes_through_data': num_passes, 'job_id':1},
        reduce=None,
        line_styles=['r-'],
        fill_colors=None,
        label_format=None,
        )


    plot_utils.draw_line_with_uncertainty(
        ax,
        combined_tree_measurement_grid,
        x='data_size',
        y='accuracy',
        slice={'number_of_passes_through_data': num_passes},
        reduce=['tree_id', 'job_id'],
        fill_color=None,
        line_style='-',
        label="trees"
        )


    # forest
    forest_measurement_grid = load_measurements(args.forest)

    plot_utils.draw_line_with_uncertainty(
        ax,
        forest_measurement_grid,
        x='data_size',
        y='accuracy',
        slice={'number_of_passes_through_data': num_passes},
        reduce=['job_id'],
        fill_color='g',
        line_style='-',
        label="forest"
        )

    # plot a "bayes" line
    x = [
        np.min(forest_measurement_grid.domain.extent['data_size']),
        np.max(forest_measurement_grid.domain.extent['data_size'])
        ]
    with open(args.forest) as pkl_file:
        data = pickle.load(pkl_file)
    y = [data['run_config'].bayes_accuracy]*2
    ax.plot(x, y, label="bayes")
    


    
    ax.legend(loc="lower right")
    ax.set_title("Forest and tree accuracy")

    
    plt.show()
