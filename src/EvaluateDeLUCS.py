# Dependencies
import os
import torch
import pickle
import numpy as np
import random
import argparse
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers import cluster_acc, plot_confusion_matrix
from PytorchUtils import Seq_data, Net_linear, IID_loss
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Random Seeds for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def eval_training(net, training_set, x_test, y_test, l=1.0, _lr=0.0001, k=6):
    """
    :param x_test: features of the new sequences to be tested
    :param y_test: "ground truth" of the sequences-optional.
    :param net: Network to be trained
    :param training_set: Dataset with pairs of CGRs of the form (original, mimic)
    :param l: hyperparameter to favor conditional entropy
    :param _lr: Learning Rate
    :param k: word length in k-mer counts.
    :return: Trained Network.
    """
    # Training parameters:
    batch_size = 512
    epochs = 150
    dtype = torch.cuda.FloatTensor

    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # -------------------Training the network-------------------------------------

    optimizer = optim.Adam(net.parameters(), lr=_lr)

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

            running_loss += loss

        running_loss /= i_batch

        if epoch % 30 == 0 and epoch != 0:
            with torch.no_grad():
                for param in net.parameters():
                    param.add_(torch.randn(param.size()).type(dtype) * 0.09)

    # ------------------- Testing Process -------------------------------------

    # Testing Parameters
    net.eval()
    predicted = []
    y_true = []  # "ground truth" is available

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

    ind, acc = cluster_acc(y_true, predicted)

    d = {}
    for i, j in ind:
        d[i] = j

    for i in range(x_test.shape[0]):  # we do this for each sample or sample batch
        predicted[i] = d[predicted[i]]

    return predicted, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', action='store', type=str, default='None')
    parser.add_argument('--out_dir', action='store', type=str, default='False')
    args = parser.parse_args()

    torch.manual_seed(0)

    # Set value of k
    k = 6

    # Load Training Data.
    data_dir = args.data_dir
    filename = os.path.join(data_dir, 'testing_data.p')

    x_train, x_test, y_test = pickle.load(open(filename, 'rb'))
    print("The size of the training array is:", x_train.shape)

    unique_labels = sorted(set(y_test))
    numClasses = len(unique_labels)

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
    predictions = []
    accuracies = []

    for i in range(10):
        l = 2.8  # 2.8
        _lr = 8.e-5

        # Initialize the network using the Kaiming technique
        net = Net_linear(4 ** k, numClasses)
        net.apply(weights_init)
        net.cuda()

        prediction, acc = eval_training(net, training_set, x_test, y_test, l=l, _lr=_lr, k=6)
        predictions.append(prediction)
        accuracies.append(acc)

    predictions = np.array(predictions)
    mode, counts = stats.mode(predictions, axis=0)
    # print(mode)
    print(accuracies)

    w = np.zeros((numClasses, numClasses), dtype=np.int64)
    for i in range(y_test.shape[0]):
        w[y_test[i], mode[0][i]] += 1

        # Print "misclassified" sequences.
        # if y_test[i] != mode[0][i]:
        #    print(i, y_test[i])

    print(w)
    print("accuracy: ", np.sum(np.diag(w) / np.sum(w)))
    PATH = os.path.join(args.out_dir, 'Confusion Matrix.png')
    plot_confusion_matrix(w, unique_labels, PATH, normalize=False)


if __name__ == '__main__':
    main()
