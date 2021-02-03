import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment


def kmer_count(seq, k):
    """
    COmpute the kmer counts for a given sequence
    :param seq:
    :param k:
    :return: Counts.
    """
    kmerDict = {}

    for k_mer in product('ACGT', repeat=k):
        kmer = ''.join(k_mer)
        kmerDict[kmer] = 0

    idx = 0

    while idx < len(seq) - k:
        try:
            kmerDict[seq[idx:idx + k]] += 1
        except KeyError:
            pass
        idx += 1

    return list(kmerDict.values())


def pos_gen(kmer):
    """
    Find the position of a particular kmer in the CGR.
    :param kmer: string with the kmer.
    :return: position in the CGR.
    """
    k = len(kmer)

    posx = 2 ** k
    posy = 2 ** k

    for i in range(1, k + 1):
        bp = kmer[-i]
        if bp == 'C':
            posx = posx - 2 ** (k - i)
            posy = posy - 2 ** (k - i)

        elif bp == 'A':
            posx = posx - 2 ** (k - i)

        elif bp == 'G':
            posy = posy - 2 ** (k - i)

    return int(posx - 1), int(posy - 1)


def cgr_gen(probs, k):
    """
    Generate CGR from the kmer counts for a given value of k.
    :param probs: array with the normalized kmer counts
    :param k:
    :return: 2D - CGR pattern.
    """
    kamers = product('ACGT', repeat=k)
    mat = np.zeros((2 ** k, 2 ** k))

    for i, kmer in enumerate(kamers):
        x, y = pos_gen(kmer)
        mat[y][x] = probs[i]

    return mat


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    :param y_true: true labels, numpy.array with shape `(n_samples,)`
    :param y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    :return:  accuracy, in [0,1]
    """

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # print(w)
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return ind, sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def plot_confusion_matrix(cm,
                          target_names,
                          PATH,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    PATH: PLace to store the image

    ---------
    Taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    ---------



    """
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(PATH)
    plt.show()
