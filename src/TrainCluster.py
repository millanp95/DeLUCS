# Dependencies

import alter
import torch
import pickle
import numpy as np
import argparse
import torch.optim as optim
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from helpers import getStats, cluster_acc, kmer_count, cgr_gen
from PytorchUtils import Seq_data, Net, IID_loss


def get_features(train, k, load_data='False'):
    train_features = []
    test_features = []
    test_labels = []

    unique_labels = list(set(map(lambda x: x[0], train)))
    numClasses = len(unique_labels)

    # Compute Features and save original data for testing.
    for i in range(len(train)):

        t = kmer_count(train[i][1], k)
        t = cgr_gen(t, k)
        t = normalize(t, norm='l2').astype('float32')

        # Save the original cgr for testing.
        test_features.append(t)
        labels = np.zeros(numClasses, dtype='float32')
        labels[unique_labels.index(train[i][0])] = 1
        test_labels.append(labels)

        if load_data:
            x_train = pickle.load(open('unsupervised_train', "rb"))
        else:
            # Append all the modified pairs for training.
            for j in range(1):
                t_gaussian = alter.add_noise(t)

                t_trans = alter.transition(train[i][1], 1 - 1e-4)
                t_trans = kmer_count(t_trans, k)
                t_trans = cgr_gen(t_trans, k)
                t_trans = normalize(t_trans, norm='l2')

                t_traver = alter.transversion(train[i][1], 1 - 0.5e-4)
                t_traver = kmer_count(t_traver, k)
                t_traver = cgr_gen(t_traver, k)
                t_traver = normalize(t_traver, norm='l2')

                t_mutated = alter.transversion(alter.transition(train[i][1], 1 - 1e-4), 1 - 0.5e-4)
                t_mutated = kmer_count(t_mutated, k)
                t_mutated = cgr_gen(t_mutated, k)
                t_mutated = normalize(t_mutated, norm='l2')

                train_features.extend([(t, t_trans), (t, t_traver),
                                       (t, t_mutated), (t, t_gaussian)])

                # train_features.extend([(t, t_trans), (t, t_traver), (t, t_mutated)])
            x_train = np.asarray(train_features).astype('float32')
            # Saving the mutated pairs.
            pickle.dump(x_train, open('unsupervised_train', "wb"))

    x_test = np.asarray(test_features).astype('float32')
    y_test = np.asarray(test_labels)  # True labels

    return x_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, default='None')
    parser.add_argument('--load_data', action='store', type=str, default='False')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")

    torch.manual_seed(0)

    # Set value of k
    k = 6

    # Load Training Data.

    TrainDataFile = args.data_path
    train = pickle.load(open(TrainDataFile, "rb"))

    # get some stats about training and testing dataset
    diTrain = getStats(train)

    # Create Labels for Classes
    diLabels = {}
    classId = 0
    numClasses = len(diTrain)

    for item in diTrain:
        classId += 1
        diLabels[item] = classId

    unique_labels = list(set(map(lambda x: x[0], train)))
    numClasses = len(unique_labels)

    x_train, x_test, y_test = get_features(train, k, args.load_data)

    print("The size of the training array is:", x_train.shape)

    # creating the dataset.
    training_set = Seq_data(x_train)

    # Building the network.
    kernel_size = 3
    net = Net(2 ** k, numClasses, 3)
    net = net.cuda()
    print(net)

    # ---- IIC training process. ---------

    # Training parameters:
    batch_size = 50
    epochs = 1000
    unique_subtypes = unique_labels
    n_classes = len(unique_subtypes)
    dtype = torch.cuda.FloatTensor
    # dtype = torch.FloatTensor

    # net = CNN( max_length, n_classes, kernel_size)

    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # -------------------Training the network-------------------------------------
    print('----------Summary of the Training Process-------------------')

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    n_samples = x_train.shape[0]
    training_losses = []
    training_acc = []

    for epoch in range(epochs):  # This is the number of times we want to iterate over the full dataset
        running_loss = 0.0
        net.train()

        for i_batch, sample_batched in enumerate(dataloader):
            sample = sample_batched['true'].view(-1, 1, 2 ** k, 2 ** k).type(dtype)
            modified_sample = sample_batched['modified'].view(-1, 1, 2 ** k, 2 ** k).type(dtype)

            # zero the gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            z1 = net(sample)
            z2 = net(modified_sample)

            loss = IID_loss(z1, z2)
            loss.backward()
            optimizer.step()

            running_loss += loss

        running_loss /= i_batch
        training_losses.append(running_loss)

        # Compute the ACC and save the best assignment.
        net.eval()
        predicted = []
        y_true = []
        # We need to optimize the testing part.
        for i in range(x_test.shape[0]):  # we do this for each sample or sample batch

            sample = torch.from_numpy(x_test[i])
            label = np.argmax(y_test[i])

            sample = sample.view(1, 1, 2 ** k, 2 ** k).type(dtype)
            output = net(sample)

            top_n, top_i = output.topk(1)  # Get Label from prediction.

            predicted.append(top_i[0].item())
            y_true.append(label)

        predicted = np.array(predicted)
        y_true = np.array(y_true)

        acc = cluster_acc(y_true, predicted)
        training_acc.append(acc)

        if acc >= max(training_acc):
            torch.save(net.state_dict(), "best_net.pytorch")

        if epoch % 25 == 0:
            print("Epoch: %s   Loss: %s --  Accuracy: %s" % (epoch, running_loss, acc))

        print('-----------Finished Training-------------')

if __name__ == '__main__':
    main()
