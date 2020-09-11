# Dependencies
import os
import torch
import pickle
import numpy as np
import argparse
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers import cluster_acc, getStats
from PytorchUtils import Seq_data, Net_linear, IID_loss
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def eval_training(net, training_set, x_test, y_test, l=1.0, _lr=0.0001, k=6):
    # Training parameters:
    batch_size = 512
    epochs = 150
    dtype = torch.cuda.FloatTensor

    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # -------------------Training the network-------------------------------------

    optimizer = optim.Adam(net.parameters(), lr=_lr)
    max_acc = 0.0

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

            loss = IID_loss(z1, z2, lamb=l)
            loss.backward()
            optimizer.step()

        # Compute the ACC and save the best assignment.
        net.eval()
        predicted = []
        y_true = []

        # We need to optimize the testing part.
        for i in range(x_test.shape[0]):  # we do this for each sample or sample batch

            sample = torch.from_numpy(x_test[i])
            label = y_test[i]

            sample = sample.view(1, 1, 2 ** k, 2 ** k).type(dtype)
            output = net(sample)

            top_n, top_i = output.topk(1)  # Get Label from prediction.

            predicted.append(top_i[0].item())
            y_true.append(label)

        predicted = np.array(predicted)
        y_true = np.array(y_true)

        acc = cluster_acc(y_true, predicted)

        if epoch % 50 == 0 and epoch != 0:
            with torch.no_grad():
                for param in net.parameters():
                    param.add_(torch.randn(param.size()).type(dtype) * 0.09)

        if acc >= max_acc:
            max_acc = acc

    return max_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', type=str, default='None')
    parser.add_argument('--pairs', action='store', type=str, default='None')  # [noise,mutation]
    args = parser.parse_args()

    print(args.data)

    # Set value of k
    k = 6

    # Load Training Data.
    filename = args.pairs
    TrainDataFile = args.data

    data = pickle.load(open(TrainDataFile, "rb"))
    unique_labels = sorted(set(map(lambda x: x[0], data)))
    numClasses = len(unique_labels)
    diData = getStats(data)

    x_train, x_test, y_test = pickle.load(open(filename, 'rb'))
    print("The size of the training array is:", x_train.shape)

    # scaling the data.
    scaler = StandardScaler()
    scaler.fit(x_test)

    x_train_1 = scaler.transform(x_train[:, 0, :])
    x_train_2 = scaler.transform(x_train[:, 1, :])
    x_test = scaler.transform(x_test)

    x_train[:, 0, :] = x_train_1
    x_train[:, 1, :] = x_train_2

    # creating the dataset.
    training_set = Seq_data(x_train)

    LAMBDA = []
    _LR = []
    accuracy = []
    LR = np.logspace(-6, -2, 1000)

    for n in range(30):

        l = random.uniform(1, 3)
        _lr = np.random.choice(LR)
        net = Net_linear(4 ** k, numClasses)
        net.apply(weights_init)
        net.cuda()

        acc = eval_training(net, training_set, x_test, y_test, l=l, _lr=_lr, k=6)

        accuracy.append(acc)
        LAMBDA.append(l)
        _LR.append(_lr)

        print(n, l, _lr, acc)


if __name__ == '__main__':
    main()
