import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from scipy.optimize import linear_sum_assignment


# extract classess from a list of tuples class-genome

def getClasses(li):
    classes = set()

    for i in range(len(li)):
        classId = li[i][0]
        classes.add(classId)

    return classes


# get class occurence stats

def getStats(li):
    classes = getClasses(li)
    diClasses = {}

    for item in classes:
        diClasses[item] = 0

    tot = 0
    maxLen = 0
    for i in range(len(li)):
        diClasses[li[i][0]] += 1
        tot += 1

        if len(li[i][1]) > maxLen:
            maxLen = len(li[i][1])

    print("-----------some stats: ------------")
    print("total num of classes: ", len(diClasses))
    print("total num of samples: ", tot)
    print("max genome length: ", maxLen)
    print("occurences: ")
    for key, item in diClasses.items():
        print(f'{key:8} => {item:8}')
    print("-----------------------------------")

    return diClasses


# plot histogram of class occurrences

def plotDict(d, path, title):
    plt.bar(range(len(d)), d.values(), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.xticks(rotation=90)
    plt.title(title)
    # plt.savefig(path)
    plt.show()


# convert letters to numbers

def genome2tabInt(genome, maxLen):
    tabInt = np.zeros(maxLen)

    # please change this later.

    for i in range(len(genome) - 1):
        if ord(genome[i]) != 32:
            tabInt[i] = ord(genome[i])

    return tabInt


# plot confusion matrix

def plot_confusion_matrix(cm,
                          target_names,
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

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
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
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def kmer_count(seq, k):
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


# Theoretically we should have 4^k features but some k-mers might not be present
# ---------------------------------
def avg_num_nonzero_entries(features):
    return int(sum(np.count_nonzero(f) for f in features) / len(features))


# This is the machine Learning Pipeline, taken from Kameris
# -----------------------------------------------
def build_pipeline(num_features):
    normalize_features = True
    dim_reduce_fraction = 0.1

    # setup normalizers if needed
    normalizers = []

    normalizers.append(('scaler', StandardScaler(with_mean=False)))

    # reduce dimensionality to some fraction of its original
    normalizers.append(('dim_reducer', TruncatedSVD(n_components=int(
        np.ceil(num_features * dim_reduce_fraction)))))

    # Classifier
    normalizers.append(('classifier', svm.SVC(kernel='linear')))

    return Pipeline(normalizers)


# This functions are for getting the number of linear features after flattening\

def size_conv1d(L_in, kernel_size, stride=1, padding=0, dilation=1):
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return L_out


def size_flatten(signal_length, n_layers, kernel_size, pooling_size, stride=1, padding=0, dilation=1):
    x = signal_length

    for i in range(n_layers):
        x = size_conv1d(x, kernel_size, stride)
        x = size_conv1d(int(x), pooling_size, pooling_size)

    return (int(x))


# This pair of function take a sequence and build the cgr
# this function calls the kmer count one.
# -------------------------------------------------------

def pos_gen(kmer):
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
    kamers = product('ACGT', repeat=k)
    mat = np.zeros((2 ** k, 2 ** k))

    for i, kmer in enumerate(kamers):
        x, y = pos_gen(kmer)
        mat[y][x] = probs[i]

    return mat


def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence), 4), dtype=np.int8)
    genome2one_hot(zeros_array=to_return, sequence=sequence, one_hot_axis=1)
    return to_return


def genome2one_hot(sequence):
    one_hot = np.zeros((len(sequence), 4), dtype=np.int8)

    for (i, char) in enumerate(sequence):
        if (char == "A" or char == "a"):
            char_idx = 0
        elif (char == "C" or char == "c"):
            char_idx = 1
        elif (char == "G" or char == "g"):
            char_idx = 2
        elif (char == "T" or char == "t"):
            char_idx = 3
        elif (char == "N" or char == "n"):
            continue  # leave that pos as all 0's
        else:
            continue

        one_hot[i, char_idx] = 1

    return one_hot


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
