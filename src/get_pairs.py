import numpy as np
import pickle
import argparse
import mimics
from itertools import product
from helpers import kmer_count


def modify_mutation(train, k, n_mimics, output_path):
    """
    :param train: dataset in pickle format
    :param k: word length
    :param output_path: file to save the mimics.
    :return: Pickle file with mimic sequences.
    """

    unique_labels = sorted(set(map(lambda x: x[0], train)))

    train_features = []
    test_features = []
    test_labels = []

    # Compute Features and save original data for testing.
    print('............computing learning pairs................')

    kmer_dict = {}
    idx = 0
    for k_mer in product('ACGT', repeat=k):
        kmer = ''.join(k_mer)
        kmer_dict[kmer] = idx
        idx += 1

    for i in range(len(train)):

        t = kmer_count(train[i][1], k)
        t_norm = t / np.sum(t)
        # Save the original cgr for testing.
        test_features.append(t_norm)
        label = unique_labels.index(train[i][0])
        test_labels.append(label)

        # Append all the modified pairs for training.

        indices, mutations = mimics.transition(train[i][1], 1 - 1e-4)
        t_trans = mimics.mutate_kmers(train[i][1], kmer_dict, t, k, indices, mutations)
        t_trans = t_trans / np.sum(t_trans)

        indices, mutations = mimics.transversion(train[i][1], 1 - 0.5e-4)
        t_traver = mimics.mutate_kmers(train[i][1], kmer_dict, t, k, indices, mutations)
        t_traver = t_traver / np.sum(t_traver)

        for j in range(n_mimics-2):
            indices, mutations = mimics.transition_transversion(train[i][1], 1 - 1e-4, 1 - 0.5e-4)
            t_mutated = mimics.mutate_kmers(train[i][1], kmer_dict, t, k, indices, mutations)
            t_mutated = t_mutated / np.sum(t_mutated)

            train_features.extend([(t_norm, t_trans), (t_norm, t_traver), (t_norm, t_mutated)])

    x_train = np.array(train_features).astype('float32')
    x_test = np.array(test_features).astype('float32')
    y_test = np.array(test_labels)  # True labels

    # Save data
    print('......saving mutated pairs.....')
    data = (x_train, x_test, y_test)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def modify_noise(train, k, n_mimics, output_path):
    """
    :param train: dataset in pickle format
    :param k: word length
    :param output_path: file to save the mimics.
    :return: Pickle file with mimic sequences.
    """
    unique_labels = sorted(set(map(lambda x: x[0], train)))

    train_features = []
    test_features = []
    test_labels = []
    n_pairs = 3

    # Compute Features and save original data for testing.
    print('Computing training pairs')
    for i in range(len(train)):
        t = kmer_count(train[i][1], k)
        t = t / np.sum(t)

        # Save the original cgr for testing.
        test_features.append(t)
        label = unique_labels.index(train[i][0])
        test_labels.append(label)

        # Append all the modified pairs for training.
        for j in range(n_pairs):
            t_gaussian = mimics.add_noise(t)
            train_features.extend([(t, t_gaussian)])

    x_train = np.asarray(train_features).astype('float32')
    x_test = np.asarray(test_features).astype('float32')
    y_test = np.asarray(test_labels)  # True labels

    print('.....saving data.....')
    # Save data
    data = (x_train, x_test, y_test)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, default='None')
    parser.add_argument('--k', action='store', type=int, default=6)
    parser.add_argument('--n_mimics', action='store', type=int, default=3)
    parser.add_argument('--modify', action='store', type=str, default='mutation')  # [noise,mutation]
    parser.add_argument('--output', action='store', type=str, default=None)

    args = parser.parse_args()

    TrainDataFile = args.data_path
    data = pickle.load(open(TrainDataFile, "rb"))

    k = args.k

    if args.modify == 'mutation':
        modify_mutation(data, k, args.n_mimics, args.output)
    elif args.modify == 'noise':
        modify_noise(data, k, args.n_mimics, args.output)
    else:
        print('Unsupported modification method')


if __name__ == '__main__':
    main()
