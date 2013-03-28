import numpy as np
import cPickle as pickle
import argparse
import matplotlib.pyplot as plt


def load_data(input_folders, sample_data, run_offset, sample_multiplier):
    x_axis = [(run+run_offset)*num_samples*sample_multiplier for (run, num_samples) in sample_data]
    for (run, num_samples) in sample_data:
        print "%d %d %d" % (run, num_samples, sample_multiplier)
    data_matrix = np.zeros((len(input_folders), len(x_axis)))

    for (i, (run, num_samples)) in enumerate(sample_data):
        for j, folder in enumerate(input_folders):
            pickle_filename = "%s/accuracy-%d-%d.pkl" % (folder, run, num_samples)
            accuracy = pickle.load(open(pickle_filename, 'rb'))
            data_matrix[j, i] = accuracy
    return x_axis, data_matrix


def plot_line(x_axis, data, line_type, color, label, plot_standard_deviation=True):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    plt.plot(x_axis, means, line_type, lw=2, color=color, label=label)
    if plot_standard_deviation:
        plt.fill_between(x_axis, means-stds, means+stds, alpha=0.2, color=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forest performance')
    parser.add_argument('-o', '--out_plot_file', type=str, required=True)
    args = parser.parse_args()

    online_samples = [(0,100), (0, 200), (0, 500), (0, 1000), (0, 2000), (0, 5000),
                        (0, 10000), (0, 25000), (0, 50000), (0, 100000), (0, 250000), (0, 500000),
                        (0,973909),(1,973909),(2,973909)]

    online_samples = [(0,100), (0, 200), (0, 500), (0, 1000)]

    # folders = ['/media/data/projects/rftk-3/icml_kinect/experiment_data_online_iid/online-iid-tree-25-n-1000-m-10-splitrate-1.00-splitroot-10.00-evalperiod-5-maxdepth-500-2013-03-28-14-33-24.598709',
    #             '/media/data/projects/rftk/icml_kinect/experiment_data_online_iid/online-iid-tree-25-n-1000-m-10-splitrate-1.00-splitroot-10.00-evalperiod-5-maxdepth-500-2013-03-28-14-37-50.788713']
    # x_axis, d = load_data(folders, online_samples,  1, 1)
    # plot_line(x_axis=x_axis, data=d, line_type='-', color='b', label='Combined')

    folders = ['/media/data/projects/rftk-3/icml_kinect/experiment_data_online_iid/online-iid-tree-25-n-1000-m-10-splitrate-1.00-splitroot-10.00-evalperiod-5-maxdepth-500-2013-03-28-14-33-24.598709']
    x_axis, d = load_data(folders, online_samples,  1, 1)
    plot_line(x_axis=x_axis, data=d, line_type='-', color='g', label='uniform random')

    folders = ['/media/data/projects/rftk/icml_kinect/experiment_data_online_iid/online-iid-tree-25-n-1000-m-10-splitrate-1.00-splitroot-10.00-evalperiod-5-maxdepth-500-2013-03-28-14-37-50.788713']
    x_axis, d = load_data(folders, online_samples,  1, 1)
    plot_line(x_axis=x_axis, data=d, line_type='-', color='r', label='at points')

    plt.title('Forest Accuracy')
    plt.xlabel('Number of sampled pixels (1000 per image)')
    plt.ylabel('Accuracy')

    plt.legend(loc = (0.5, 0.05))
    plt.savefig(args.out_plot_file)
    plt.show()
