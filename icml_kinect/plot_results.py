import numpy as np
import cPickle as pickle
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot online vs offline random forests for kinect')
    parser.add_argument('-i', '--in_results_file', type=str, required=True)
    parser.add_argument('-o', '--out_plot_file', type=str, required=True)
    args = parser.parse_args()

    results = pickle.load(file(args.in_results_file, 'rb'))
    plt.plot(results['number_of_data'], results['online_one_pass'], '-', lw=2, color='b', label='online_one_pass')
    plt.plot(results['number_of_data'], results['online_multi_pass'], '-', lw=2, color='g', label='online_multi_pass')
    plt.plot(results['number_of_data'], results['alpha_online_one_pass_accuracy'], '-.', lw=2, color='b', label='ab_online_one_pass_accuracy')
    plt.plot(results['number_of_data'], results['alpha_online_multi_pass_accuracy'], '-.', lw=2, color='g', label='ab_online_multi_pass_accuracy')
    plt.plot(results['number_of_data'], results['offline'], '-', lw=2, color='r', label='offline')

    plt.title('Kinect Online vs Offline')
    plt.xlabel('Number of images')
    plt.ylabel('Accuracy')

    plt.legend(loc = (0.5, 0.05))
    plt.savefig(args.out_plot_file)
    plt.show()