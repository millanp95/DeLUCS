"""
Toy implementation for Deep Embedded Clustering as described in the paper:

        Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

Main differences with original code at https://github.com/piiswrong/dec.git:
    1. Autoencoder is pretrain in an end-to-end manner, while original code is in greedy layer-wise training manner.

Usage:
    No pretrained autoencoder weights available:
        python DEC.py mnist
        python DEC.py usps
        python DEC.py reutersidf10k --n_clusters 4
    Weights of Pretrained autoencoder for mnist are in './ae_weights/mnist_ae_weights.h5':
        python DEC.py mnist --ae_weights ./ae_weights/mnist_ae_weights.h5

Author:
    Xifeng Guo. 2017.4.30
"""

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

from sklearn.cluster import KMeans
from sklearn import metrics


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
    # print(y_true.shape,y_pred.shape)
    # print(y_true[5:10,:])
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder = autoencoder(self.dims)
        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input, outputs=clustering_layer)

        self.pretrained = False
        self.centers = []
        self.y_pred = []

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam'):
        print('...Pretraining...')
        self.autoencoder.compile(loss='mse', optimizer=optimizer)  # SGD(lr=0.01, momentum=0.9),
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        self.autoencoder.save_weights('ae_weights.h5')
        print('Pretrained weights are saved to ./ae_weights.h5')
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of DEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss='kld', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None, save_dir='./results/dec'):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: pretrain
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L'])
        logwriter.writeheader()

        loss = 0
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(cluster_acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logwriter.writerow(dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss))
                    print('Iter-%d: ACC= %.4f, NMI= %.4f, ARI= %.4f;  L= %.5f'
                          % (ite, acc, nmi, ari, loss))

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=p[index * batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=p[index * batch_size:(index + 1) * batch_size])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DEC model checkpoints
                print('saving model to: ' + save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to: ' + save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return self.y_pred


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'reutersidf10k', 'pendigits','cgr'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=200, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=0, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/dec')
    parser.add_argument('--data_path', default='data/cgr/train.p')
    args = parser.parse_args()
    print(args)

    # load dataset
    optimizer = 'adam'  # SGD(lr=0.01, momentum=0.99)
    from datasets import load_mnist, load_reuters, load_usps, load_pendigits, load_cgr

    if args.dataset == 'mnist':  # recommends: n_clusters=10, update_interval=140
        x, y = load_mnist()
    elif args.dataset == 'usps':  # recommends: n_clusters=10, update_interval=30
        x, y = load_usps('data/usps')
    elif args.dataset == 'pendigits':
        x, y = load_pendigits('data/pendigits')
    elif args.dataset == 'reutersidf10k':  # recommends: n_clusters=4, update_interval=20
        x, y = load_reuters('data/reuters')
    elif args.dataset == 'cgr': # Don't forget to specify numer of clusters
            x, y = load_cgr(args.data_path)

    if args.update_interval == 0:  # one epoch. A smaller value may correspond to higher performance
        args.update_interval = int(x.shape[0]/args.batch_size)

    # Define DEC model
    dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=args.n_clusters)
    plot_model(dec.model, to_file='dec_model.png', show_shapes=True)
    dec.model.summary()

    t0 = time()

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        dec.pretrain(x, batch_size=args.batch_size, epochs=args.pretrain_epochs, optimizer=optimizer)

    # begin clustering, time not include pretraining part.

    dec.compile(loss='kld', optimizer=optimizer)
    dec.fit(x, y=y, batch_size=args.batch_size, tol=args.tol, maxiter=args.maxiter,
            update_interval=args.update_interval, ae_weights=args.ae_weights, save_dir=args.save_dir)

    # Show the final results
    y_pred = dec.y_pred
    print('acc:', cluster_acc(y, y_pred))
    print('clustering time: %d seconds.' % int(time() - t0))
