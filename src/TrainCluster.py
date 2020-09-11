# Dependencies
import os
import torch
import pickle
import numpy as np
import argparse
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers import cluster_acc
from PytorchUtils import Seq_data, Net_linear, IID_loss
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)



def eval_training(net, training_set, x_test, y_test, output_dir, l=1.0, _lr=0.0001, k=6):
    # Training parameters:
    batch_size = 64
    epochs = 100
    dtype = torch.cuda.FloatTensor

    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # -------------------Training the network-------------------------------------

    optimizer = optim.Adam(net.parameters(), lr=_lr)
    training_losses = []
    training_acc = []

    for epoch in range(epochs):  # This is the number of times we want to iterate over the full dataset
        running_loss = 0.0
        net.train()

        for i_batch, sample_batched in enumerate(dataloader):
            sample = sample_batched['true'].view(-1, 1, 2**k, 2**k).type(dtype)
            modified_sample = sample_batched['modified'].view(-1, 1, 2**k, 2**k).type(dtype)

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
        training_losses.append(running_loss)

        # Compute the ACC and save the best assignment.
        net.eval()
        predicted = []
        y_true = []

        # We need to optimize the testing part.
        for i in range(x_test.shape[0]):  # we do this for each sample or sample batch

            sample = torch.from_numpy(x_test[i])
            label = y_test[i]

            sample = sample.view(1, 1, 2**k, 2**k).type(dtype)
            output = net(sample)

            top_n, top_i = output.topk(1)  # Get Label from prediction.

            predicted.append(top_i[0].item())
            y_true.append(label)

        predicted = np.array(predicted)
        y_true = np.array(y_true)

        acc = cluster_acc(y_true, predicted)
        training_acc.append(acc)

        if acc >= max(training_acc):
            model_path = os.path.join(output_dir, "best_net.pytorch")
            torch.save(net.state_dict(), model_path)

        print("Epoch: %s   Loss: %s --  Accuracy: %s" % (epoch, running_loss, acc))

        if epoch % 50 == 0 and epoch != 0:
            with torch.no_grad():
                for param in net.parameters():
                    param.add_(torch.randn(param.size()).type(dtype) * 0.09)

        print('-----------Finished Training-------------')

    plt.plot(training_losses)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Training Process')
    figure_path = os.path.join(output_dir, 'training process.png')
    plt.savefig(figure_path)


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
    filename = os.path.join(data_dir, 'testing_data')
    TrainDataFile = os.path.join(data_dir, data_dir + '.p')

    data = pickle.load(open(TrainDataFile, "rb"))
    unique_labels = sorted(set(map(lambda x: x[0], data)))
    numClasses = len(unique_labels)

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

    # Building the network.
    net = Net_linear(4 ** k, numClasses)
    net.apply(weights_init)
    net = net.cuda()

    eval_training(net, training_set, x_test, y_test, args.out_dir, l=1.0, _lr=0.0001, k=6)

if __name__ == '__main__':
    main()
