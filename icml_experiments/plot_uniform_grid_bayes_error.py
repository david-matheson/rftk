import numpy as np
import cPickle as pickle
import argparse
import matplotlib.pyplot as plt

def plot_line_with_std(xs, ymatrix, color, label):
    means = np.mean( ymatrix, axis=0)
    stds = np.std( ymatrix, axis=0)

    plt.plot(xs, means, '-', lw=2, color=color, label=label)
    plt.fill_between(xs, means-stds, means+stds, alpha=0.2, color=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot bayes error')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-o', '--out_plot_file', type=str, required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)
    data_config = config.get_data_config()
    data_config = config.get_data_config()

    split1 = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-1/error_from_bayes.pkl', 'rb')))
    split1a = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-1a/error_from_bayes.pkl', 'rb')))
    split1b = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-1b/error_from_bayes.pkl', 'rb')))
    split1c = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-1c/error_from_bayes.pkl', 'rb')))
    split1d = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-1c/error_from_bayes.pkl', 'rb')))

    split2 = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-2/error_from_bayes.pkl', 'rb')))
    split2a = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-2a/error_from_bayes.pkl', 'rb')))
    split2b = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-2b/error_from_bayes.pkl', 'rb')))
    split2c = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-2c/error_from_bayes.pkl', 'rb')))
    split2d = np.array(pickle.load(file('experiment_data/epsilon_uniform-online-sequential-forest-100-split-2d/error_from_bayes.pkl', 'rb')))

    split1 = np.vstack((split1, split1a, split1b, split1c, split1d))
    split2 = np.vstack((split2, split2a, split2b, split2c, split2d))

    plot_line_with_std(data_config.data_sizes, split1, color='b', label='split rate of 1.0')
    plot_line_with_std(data_config.data_sizes, split2, color='g', label='split rate of 2.0')


    plt.title('Gap to Bayes')
    plt.xlabel('Data Size')
    plt.xlim([150,1000000])
    plt.ylabel('Bayes Error')
    plt.ylim([0,600])
    # plt.get_yaxis().set_ticks([])
    plt.xscale('log')

    plt.legend(loc = (0.6, 0.80))
    plt.savefig(args.out_plot_file)
    plt.show()