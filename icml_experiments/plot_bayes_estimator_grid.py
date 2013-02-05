import argparse
import cPickle as pickle

import plot_utils
import dist_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Synthetic')
    parser.add_argument('-d', '--distribution', default="5-class", help='distribution name')
    parser.add_argument('--n_train', type=int, help='number of training points', required=True)
    parser.add_argument('--n_test', type=int, help='number of test points', required=True)
    parser.add_argument('-o', '--out_file', default="data_out.pkl", help='dataset file')
    args = parser.parse_args()

    dist = dist_utils.get_mog_dist(args.distribution)
    X_train, Y_train = dist.sample(args.n_train)
    X_test, Y_test = dist.sample(args.n_test)

    pickle.dump((X_train, Y_train, X_test, Y_test), open(args.out_file))