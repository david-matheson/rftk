import cPickle as pickle
import argparse
import numpy as np
import dist_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mixture of gaussian dataset')
    parser.add_argument('-d', '--distribution', default="3-class", help='distribution name')
    parser.add_argument('--number_of_train', type=int, required=True)
    parser.add_argument('--number_of_test', type=int, required=True)
    parser.add_argument('-o', '--out_file', default="data.pkl")
    args = parser.parse_args()

    dist = dist_utils.get_mog_dist(args.distribution)
    X_train, Y_train = dist.sample(args.number_of_train)
    X_test, Y_test = dist.sample(args.number_of_test)

    pickle.dump((X_train, Y_train, X_test, Y_test), open(args.out_file, "wb"))

    # Bayes accuracy
    y_probs = dist.predict_proba(X_test)
    y_hat = y_probs.argmax(axis=1)
    bayes_accuracy = np.mean(Y_test == y_hat)
    print bayes_accuracy
