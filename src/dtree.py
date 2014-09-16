import mldata as md
import numpy as np
import argparse


def main(problem_name, max_depth):
    dataset = md.parse_c45(problem_name)
    data_array = np.array(dataset.to_float())
    np.random.shuffle(data_array)
    training_set = data_array[:4 * len(data_array)/5]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of the ID3 algorithm.')
    parser.add_argument('problem_name', metavar='P',
                   help='The name of the problem set to process.')
    parser.add_argument('--max-depth', action='store_const',
                   help='Limit the tree to a certain depth.')

    args = parser.parse_args()
    main(args.problem_name, args.max_depth)