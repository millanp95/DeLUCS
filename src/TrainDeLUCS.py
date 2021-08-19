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


def training(net, training_set, l=1.0, _lr=0.0001, k=6):
    """
    :param net: Network to be trained
    :param training_set: Dataset with pairs of CGRs of the form (original, mimic)
    :param l: Hyperparameter to favor conditional entropy
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


def predict(net, x_test, k=6):
    """
    Test the model for a new dataset.
    :param net: Trained Network.
    :param x_test: features of the new sequences to be tested
    :param k: word length for the k-mer counts
    :return: classification and clustering accuracy
    """

    dtype = torch.cuda.FloatTensor
    net.eval()
    predicted = []

    for i in range(x_test.shape[0]):  # we do this for each sample or sample batch

        sample = torch.from_numpy(x_test[i])
        sample = sample.view(1, 1, 2 ** k, 2 ** k).type(dtype)
        output = net(sample)
        top_n, top_i = output.topk(1)  # Get Label from prediction.
        predicted.append(top_i[0].item())

    predicted = np.array(predicted)

    # NO computation of the Hungarian.
    # we return just the predictions.

    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', action='store', type=str, default='None')
    parser.add_argument('--out_dir', action='store', type=str, default='False')
    parser.add_argument('--n_custers', action='store', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)

    # Set value of k
    k = 6

    # Load Training Data.
    data_dir = args.data_dir
    filename = os.path.join(data_dir, 'testing_data.p')

    x_train, x_test, y_test = pickle.load(open(filename, 'rb'))
    print("The size of the training array is:", x_train.shape)

    if args.n_clusters == 0:
        unique_labels = sorted(set(y_test))
        numClasses = len(unique_labels)
    else:
        numClasses = args.n_clusters

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

    for i in range(10):
        l = 2.8  # 2.8
        _lr = 8.e-5

        # Initialize the network using the Kaiming technique
        net = Net_linear(4 ** k, numClasses)
        net.apply(weights_init)
        net.cuda()

        training(net, training_set, l=l, _lr=_lr, k=6)
        prediction = predict(net, x_test)
        predictions.append(prediction)

    # There is no ground truth. Hence, we find a uniform
    # assignment to all predictions.
    predictions = np.array(predictions)
    for k in range(1, predictions.shape[0]):
        ind, _ = cluster_acc(predictions[0][:], predictions[:][k])

        d = {}
        for i, j in ind:
            d[i] = j

        for i in range(x_test.shape[0]):
            predictions[k][i] = d[predictions[k][i]]

    # Take the majority voting of the predictions.
    mode, counts = stats.mode(predictions, axis=0)

    # Load the original file to get the original sequences.
    original_files = os.path.join(data_dir, 'train.p')
    data = pickle.load(open(original_files, "rb"))
    prediction_pairs = []

    # Assign the prediction to each accession number.
    for i in range(len(data)):
        prediction_pairs.append((data[i][2], mode[0][i]))
        print((data[i][2], mode[0][i]))

    # Save the final prediction
    PATH = os.path.join(args.out_dir, 'predictions.p')
    pickle.dump(prediction_pairs, open(PATH, "wb"))


if __name__ == '__main__':
    main()
