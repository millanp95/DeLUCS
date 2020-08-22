import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize


# ------ add_noise ----------------

def add_noise(x_train):
    n_features = x_train.shape[0]
    index = (np.random.random(n_features) < 0.25).astype('float32')
    noise = np.random.normal(0, 0.001, n_features)
    gaussian_train = x_train + noise * index
    return gaussian_train


# ----- Transition -----------------
def transition(seq, threshold):
    mutated_seq = []
    for nucleotide in seq:
        prob = np.random.uniform()
        if prob > threshold:
            if nucleotide == 'A':
                mutated_seq.append('G')
            if nucleotide == 'G':
                mutated_seq.append('A')
            if nucleotide == 'T':
                mutated_seq.append('C')
            if nucleotide == 'C':
                mutated_seq.append('T')
        else:
            mutated_seq.append(nucleotide)
    l = len(mutated_seq)
    mutated_seq += ' ' * (len(seq) - l)
    return ''.join(mutated_seq)


# ------ Transversion ---------------

def transversion(seq, threshold):
    mutated_seq = []
    for nucleotide in seq:
        prob = np.random.uniform()
        if prob > threshold:
            if nucleotide == 'A':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutated_seq.append('T')
                else:
                    mutated_seq.append('C')
            if nucleotide == 'G':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutated_seq.append('T')
                else:
                    mutated_seq.append('C')
            if nucleotide == 'T':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutated_seq.append('A')
                else:
                    mutated_seq.append('G')
            if nucleotide == 'C':
                random_number = np.random.uniform()
                if random_number > 0.5:
                    mutated_seq.append('A')
                else:
                    mutated_seq.append('G')
        else:
            mutated_seq.append(nucleotide)
    l = len(mutated_seq)
    mutated_seq += ' ' * (len(seq) - l)
    return ''.join(mutated_seq)
