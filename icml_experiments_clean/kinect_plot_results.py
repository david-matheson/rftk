import argparse
import glob
import numpy as np
import cPickle as pickle
import matplotlib
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
    parser = argparse.ArgumentParser(description='Plot forest performance')
    parser.add_argument('--list_of_samples_to_plot', type=str, required=True)  
    parser.add_argument('--saffari_folders', type=str, required=True)   
    parser.add_argument('--online_folders', type=str, required=True)  
    parser.add_argument('--out_plot_file', type=str, required=True)
    args = parser.parse_args()

    samples_to_plot = eval(args.list_of_samples_to_plot)

    params = {
            'axes.labelsize' : 20,
            'font.size' : 20,
            'text.fontsize' : 20,
            'legend.fontsize': 20,
            'xtick.labelsize' : 15,
            'ytick.labelsize' : 15,
            }
    matplotlib.rcParams.update(params)

    saffari_folders = glob.glob(args.saffari_folders)
    print saffari_folders
    x_axis, d = load_data(saffari_folders, samples_to_plot,  1, 1)
    plot_line(x_axis=x_axis, data=d, line_type='-', color='g', label='Saffari et al. (2009)')

    online_folders = glob.glob(args.online_folders)
    print online_folders
    x_axis, d = load_data(online_folders, samples_to_plot,  1, 1)
    plot_line(x_axis=x_axis, data=d, line_type='-', color='b', label='$\\alpha(d)=10\\cdot(1.01^d)$')

    plt.title('Forest Accuracy')
    plt.xlabel('Number of sampled pixels (1000 per image)')
    plt.ylabel('Accuracy')

    plt.xscale('log')
    plt.legend(loc = (0.38, 0.05))
    plt.savefig(args.out_plot_file)
    plt.show()
