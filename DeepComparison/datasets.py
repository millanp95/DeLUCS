import numpy as np
import pickle


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)  # normalize as it does in DEC paper
    print('MNIST samples', x.shape)
    return x, y

def load_cgr(datafile):

    from helpers import kmer_count, build_pipeline, cgr_gen
    from sklearn.preprocessing import normalize

    #the data comes from a the prepare data script
    DataFile = datafile
    data = pickle.load(open(DataFile, "rb"))
    unique_labels = list(set(map(lambda x: x[0], data)))
    numClasses = len(unique_labels)

    test_features = []
    test_labels = []

    for i in range(len(data)):
        k = 6
        t = kmer_count(data[i][1],k)
        t = np.array(t)
        t = t / np.sum(t)
        test_features.append(t)

        test_labels.append(unique_labels.index(data[i][0]))

    x = np.asarray(test_features).astype('float32')
    y = np.asarray(test_labels)

    print('CGR samples', x.shape)
    return x, y

def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_pendigits(data_path='./data/pendigits'):
    import os
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tra -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tes -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.names -P %s' % data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]
    print('data_train shape=', data_train.shape)

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]
    print('data_test shape=', data_test.shape)

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    print('pendigits samples:', x.shape)
    return x, y


def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print('reutersidf saved to ' + data_path)
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float32')
    y = y.reshape((y.size,))
    print('REUTERSIDF10K samples', x.shape)
    return x, y


def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x, y = x[:10000], y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x, y = x[p], y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})
