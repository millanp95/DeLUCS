import numpy as np


def add_noise(x_train):
    """
    Add artificial Gaussian noise to a training sample.
    :param x_train: ndarray with normalized kmers.
    :return: ndarray with gaussian noise.
    """
    n_features = x_train.shape[0]
    index = (np.random.random(n_features) < 0.25).astype('float32')
    noise = np.random.normal(0, 0.001, n_features)
    gaussian_train = x_train + noise * index
    return gaussian_train


def mutate_kmers(seq, k_dict, k_count, k, positions, mutations):
    """
    Compute the k-mer counts based on mutations.

    :param seq: Original Sequence to be mutated.
    :param k_dict: Dictionary with kmers.
    :param k_count: Array with k-mer counts in the original seq.
    :param k:
    :param positions: Array with the mutated positions.
    :param mutations: Array with the correspondent mutations.
    :return: Array with kmer counts of the mutated version
    """

    new_count = k_count.copy()

    for (i, new_bp) in zip(positions, mutations):
        max_j = min(k, len(seq)-i, i)
        min_j = min(k, i)
        for j in range(1, max_j + 1):
            idx = i - min_j + j
            kmer = seq[idx: idx + k]
            new_kmer = list(kmer)
            new_kmer[-j] = new_bp
            new_kmer = ''.join(new_kmer)

            new_count[k_dict[kmer]] -= 1
            new_count[k_dict[new_kmer]] += 1

    return new_count


def transition(seq, threshold):
    """
    Mutate Genomic sequence using transitions only.
    :param seq: Original Genomic Sequence.
    :param threshold: probability of NO Transition.
    :return: Mutated Sequence.
    """
    x = np.random.random(len(seq))
    index = np.where(x > threshold)[0]
    mutations = []

    for i in index:
        nucleotide = seq[i]
        if nucleotide == 'A':
            mutations.append('G')
        if nucleotide == 'G':
            mutations.append('A')
        if nucleotide == 'T':
            mutations.append('C')
        if nucleotide == 'C':
            mutations.append('T')

    return index, mutations


def transversion(seq, threshold):
    """
    Mutate Genomic sequence using transversions only.
    :param seq: Original Genomic Sequence.
    :param threshold: Probability of NO Transversion.
    :return: Mutated Sequence.
    """

    x = np.random.random(len(seq))
    index = np.where(x > threshold)[0]
    mutations = []

    for i in index:
        nucleotide = seq[i]

        if nucleotide == 'A':
            random_number = np.random.uniform()
            if random_number > 0.5:
                mutations.append('T')
            else:
                mutations.append('C')
        if nucleotide == 'G':
            random_number = np.random.uniform()
            if random_number > 0.5:
                mutations.append('T')
            else:
                mutations.append('C')
        if nucleotide == 'T':
            random_number = np.random.uniform()
            if random_number > 0.5:
                mutations.append('A')
            else:
                mutations.append('G')
        if nucleotide == 'C':
            random_number = np.random.uniform()
            if random_number > 0.5:
                mutations.append('A')
            else:
                mutations.append('G')

    return index, mutations


def transition_transversion(seq, threshold_1, threshold_2):
    """
    Mutate Genomic sequence using transitions and transversions
    :param seq: Original Sequence.
    :param threshold_1: Probability of NO transition.
    :param threshold_2: Probability of NO transversion.
    :return:
    """
    # First transitions.
    idx, mutations = transition(seq, threshold_1)
    seq = list(seq)
    # Then transversions.
    for (i, new_bp) in zip(idx, mutations):
        seq[i] = new_bp

    return transversion(''.join(seq), threshold_2)
