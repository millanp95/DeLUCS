import torch
import pickle
import numpy as np
import argparse
from helpers import kmer_count
from sklearn.pipeline import Pipeline
from sklearn import mixture
from sklearn.cluster import KMeans
import torch.optim as optim
from helpers import cluster_acc
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from helpers import plot_confusion_matrix
from PytorchUtils import Net_linear, LabeledData


def dataDescription(data):
    clusters = set()

    for i in range(len(data)):
        clusterLabel = data[i][0]
        clusters.add(clusterLabel)

    ClusterDistribution = {}

    for item in clusters:
        ClusterDistribution[item] = 0

    tot = 0
    minLen = 1e9
    maxLen = 0
    count = 0

    for i in range(len(data)):
        ClusterDistribution[data[i][0]] += 1
        tot += 1

        count += len(data[i][1])
        maxLen = max(maxLen, len(data[i][1]))
        minLen = min(minLen, len(data[i][1]))

    print("-----------some stats: ------------")
    print("Total num of classes: ", len(ClusterDistribution))
    print("Total num of samples: ", tot)
    print("Min genome length: ", minLen)
    print("Avg genome length: ", count / tot)
    print("Max genome length: ", maxLen)
    print("Cluster Distribution: ")
    for key, item in ClusterDistribution.items():
        print(f'{key:8} => {item:8}')
    print("-----------------------------------")

    return ClusterDistribution


def build_pipeline(numClasses, method):
    normalizers = []
    if method == 'GMM':
        normalizers = [('classifier', mixture.GaussianMixture(n_components=numClasses))]
    if method == 'k-means++':
        normalizers.append(('classifier', KMeans(n_clusters=numClasses, init='k-means++', random_state=321)))
    return Pipeline(normalizers)


def Unsupervised(train, method, k):
    train_features = []
    train_labels = []
    unique_labels = sorted(list(set(map(lambda x: x[0], train))))
    numClasses = len(unique_labels)

    for i in range(len(train)):
        t = kmer_count(train[i][1], k)
        t = np.array(t)
        t = t / np.sum(t)

        train_features.append(t)

        label = unique_labels.index(train[i][0])
        train_labels.append(label)

    x_train = np.asarray(train_features).astype('float32')
    y_train = np.asarray(train_labels)

    a = []
    for i in range(10):
        pipeline = build_pipeline(numClasses, method)
        pipeline.fit(x_train)
        y_pred = pipeline.predict(x_train)
        #print("Cluster Accuracy")
        a.append(cluster_acc(y_pred, y_train)[1])
    print(a, np.mean(np.array(a)))

def train(net, training_set, k):
    # Training parameters:
    batch_size = 128
    epochs = 20
    dtype = torch.cuda.FloatTensor
    # dtype = torch.FloatTensor
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    print('----------Summary of the Training Process-------------------')

    optimizer = optim.Adadelta(net.parameters(), lr=0.1)

    training_losses = []

    for epoch in range(epochs):  # This is the number of times we want to iterate over the full dataset
        running_loss = 0.0
        net.train()

        for i_batch, sample_batched in enumerate(dataloader):
            sample = sample_batched['cgr'].view(-1, 1, 2 ** k, 2 ** k).type(dtype)
            label_tensor = sample_batched['label'].type(torch.cuda.LongTensor)

            # zero the gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(sample)
            loss = criterion(output, label_tensor)
            loss.backward()

            optimizer.step()

            running_loss += loss

        running_loss /= i_batch
        print(running_loss)
        training_losses.append(running_loss)

def Supervised(x_pairs,x,y, k):

    unique_labels = sorted(set(y))
    numClasses = len(unique_labels)

    print("The size of the training array is:", x_pairs.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # creating the dataset.
    training_set = LabeledData(x_train, y_train)

    # Building the network.
    numClasses = len(unique_labels)
    net = Net_linear(4 ** k, numClasses)

    net = net.cuda()
    print(net)

    # Training Process.
    train(net, training_set, k)

    # Testing Process.
    dtype = torch.cuda.FloatTensor
    net.eval()
    predicted = []
    y_true = []

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

    print(y_true)
    print(predicted)
    n = numClasses

    w = np.zeros((n, n), dtype=np.int64)

    for i in range(predicted.shape[0]):
        w[y_true[i], predicted[i]] += 1

    plot_confusion_matrix(w, unique_labels, normalize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, default='None')
    parser.add_argument('--method', action='store', type=str, default='k-means++')  # [GMM,k-means++,supervised]
    args = parser.parse_args()

    TrainDataFile = args.data_path
    method = args.method

    torch.manual_seed(0)

    # Set value of k
    k = 6

    # Load Training Data.
    train = pickle.load(open(TrainDataFile, "rb"))

    if method in ['GMM', 'k-means++']:
        # get some stats about training and testing dataset
        dataDescription(train)
        Unsupervised(train, method, k)

    elif method == 'Supervised':
        # Get data
        x_pairs, x, y = train
        Supervised(x_pairs, x, y, k)
    else:
        print("Wrong Method")


if __name__ == '__main__':
    main()
