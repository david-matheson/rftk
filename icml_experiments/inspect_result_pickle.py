import cPickle as pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect result packages.')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    for file_name in args.files:
        with open(file_name) as result_file:
            result_package = pickle.load(result_file)
        measurements = result_package['measurements']
        print measurements
