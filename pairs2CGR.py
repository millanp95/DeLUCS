import numpy as np
import random
import pickle
import argparse
import alter
from helpers import kmer_count
import os


def reverseComp(s):
    d = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(list(map(lambda x: d.get(x, 'N'), s))[::-1])


def doubleCGR(s, k):
    t = kmer_count(s, k) + kmer_count(reverseComp(s), k)
    return t / np.sum(t)


def modify_mutation(train, k, output_path):
    unique_labels = sorted(set(map(lambda x: x[0], train)))
    numClasses = len(unique_labels)

    train_features = []
    test_features = []
    test_labels = []

    # Compute Features and save original data for testing.
    for i in range(len(train)):

        t = doubleCGR(train[i][1], k)

        # Save the original cgr for testing.
        test_features.append(t)
        label = unique_labels.index(train[i][0])
        test_labels.append(label)

        # Append all the modified pairs for training.
        for j in range(1):
            t_trans = alter.transition(train[i][1], 1 - 1e-4)
            t_trans = doubleCGR(t_trans, k)

            t_traver = alter.transversion(train[i][1], 1 - 0.5e-4)
            t_traver = doubleCGR(t_traver, k)

            t_mutated = alter.transversion(alter.transition(train[i][1], 1 - 1e-4), 1 - 0.5e-4)
            t_mutated = doubleCGR(t_mutated, k)

            train_features.extend([(t, t_trans), (t, t_traver), (t, t_mutated)])

    x_train = np.array(train_features).astype('float32')
    x_test = np.array(test_features).astype('float32')
    y_test = np.array(test_labels)  # True labels

    # Save data
    data = (x_train, x_test, y_test)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    # Send it to compute canada.
    cmd = "sshpass -p 'Tp14P!4u' scp -r %s  pmillana@cedar.computecanada.ca:~/scratch/data/%s" % (
    output_path, output_path)
    os.system(cmd)

    # Delete the file.
    os.system("rm %s" % output_path)


def modify_noise(train, k, output_path):
    unique_labels = sorted(set(map(lambda x: x[0], train)))
    numClasses = len(unique_labels)

    train_features = []
    test_features = []
    test_labels = []
    n_pairs = 3

    # Compute Features and save original data for testing.
    for i in range(len(train)):
        t = doubleCGR(train[i][1], k)

        # Save the original cgr for testing.
        test_features.append(t)
        label = unique_labels.index(train[i][0])
        test_labels.append(label)

        # Append all the modified pairs for training.
        for j in range(n_pairs):
            t_gaussian = alter.add_noise(t)
            train_features.extend([(t, t_gaussian)])

    x_train = np.asarray(train_features).astype('float32')
    x_test = np.asarray(test_features).astype('float32')
    y_test = np.asarray(test_labels)  # True labels

    # Save data (Here we will send the data to compute canada and deleting the pairs from our system)
    data = (x_train, x_test, y_test)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, default='None')
    parser.add_argument('--k', action='store', type=int, default=6)
    parser.add_argument('--modify', action='store', type=str, default='noise')  # [noise,mutation]
    parser.add_argument('--output', action='store', type=str, default=None)
    args = parser.parse_args()

    TrainDataFile = args.data_path
    data = pickle.load(open(TrainDataFile, "rb"))

    k = args.k

    unique_labels = sorted(set(map(lambda x: x[0], data)))

    if args.modify == 'mutation':
        modify_mutation(data, k, args.output)
    elif args.modify == 'noise':
        modify_noise(data, k, args.output)
    else:
        print('Unsupported modification method')


if __name__ == '__main__':
    main()
