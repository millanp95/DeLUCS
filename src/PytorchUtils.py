import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable


# -----------Creating the Class Dataset (Unsupervised) ------------------------
class Seq_data(Dataset):
    """New dataset using pairs of CGR"""

    def __init__(self, data):
        """
        Args:
            data (numpy array): True CGR.
            copy (numpy array): Modified CGR.
        """
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'true': self.data[idx, 0, :], 'modified': self.data[idx, 0, :]}
        return sample


# -----------Creating the Class Dataset (Supervised) ------------------------
class LabeledData(Dataset):
    """New dataset using ASCII encoding"""

    def __init__(self, data, labels):
        """
        Args:
            data (numpy array): CGR representation of each sequence.
            labels (numpy array): Subtypes of each sequence.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'cgr': self.data[idx, :], 'label': self.labels[idx]}

        return sample


# -------------Building the Neural Network------------------------
class Net(nn.Module):
    def __init__(self, n_input, n_output, w):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, w),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(p=0.25)
        )
        self.flat_fts = self.get_flat_fts(n_input, self.features)
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, 512),
            nn.ReLU(),
            nn.Linear(512, n_output),  # Always check n_input here.
            nn.Softmax(dim=1)
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1, 1, in_size, in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        out = self.classifier(flat_fts)
        return out


class Net_linear(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net_linear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, n_output),  # Always check n_input here.
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 4096)
        out = self.classifier(x)
        return out


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j)
                      - lamb * torch.log(p_j)
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j
