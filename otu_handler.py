import copy
import pandas as pd
import numpy as np
from skbio.stats.composition import clr, ilr
from scipy.stats.mstats import gmean, zscore

'''
Class for handling OTU data. It generates samples and keeps track of
training and validation data.
'''
class OTUHandler(object):
    def __init__(self, files):
        self.samples = []
        for f in files:
            self.samples.append(pd.read_csv(f, index_col=0))
        # Keep track of these for getting back and forth from normalized.
        self.raw_samples = copy.deepcopy(self.samples)
        self.strains = list(self.samples[0].index.values)
        self.num_strains = len(self.strains)
        self.train_data = None
        self.val_data = None
        self.normalization_method = None
        self.normalization_factors = {}

    '''
    Set the training and validation data for each sample. Can include
    a lower bound on size of train/validation
    '''
    def set_train_val(self, percent=0.8, minsize=12):
        self.train_data = []
        self.val_data = []
        self.test_data = []  # TODO: implement test data.
        temp_sizes = []
        for sample in self.samples:
            index = int(percent * sample.shape[1])
            # If not enough examples, skip this file.
            if not (len(sample.iloc[:, :index]) < minsize or
                    len(sample.iloc[:, index:]) < minsize):
                self.train_data.append(sample.iloc[:, :index])
                self.val_data.append(sample.iloc[:, index:])
                temp_sizes.append(sample.iloc[:, :index].shape[1])
                temp_sizes.append(sample.iloc[:, index:].shape[1])

        # For keeping track of max size the slice can be.
        self.min_len = min(temp_sizes)

    '''
    Method for normalizing the input data. it also keeps track of the
    relevant parameters used for normalization so that they can be
    returned to their raw values for transformation of the predicted data.
    '''
    def normalize_data(self, method='zscore'):
        method = method.lower()
        if method not in ['zscore', 'clr']:
            raise ValueError('Specify "zscore" or "clr" for method')
        if method == 'zscore':
            self.normalization_method = 'zscore'
            m = zscore
            means = np.mean
            std = np.std
        else:
            self.normalization_method = 'clr'
            m = clr
            means = gmean
            std = None
        new_vals = []
        self.normalization_factors[method] = {'mean': [],
                                              'std': []}
        for s in self.samples:
            self.normalization_factors[method]['mean'].append(means(s.values,
                                                                    axis=1))
            if std is not None:
                self.normalization_factors[method]['std'].append(std(s.values,
                                                                     axis=1))
            new_vals.append(pd.DataFrame(m(s.values),
                                         index=s.index,
                                         columns=s.columns))

        self.samples = new_vals
        # Reassign the train and test values given the normalization.
        self.set_train_val()

    '''
    Function for returning the normalized values to the raw values.
    This is good for plotting the predicted values versus actual values.
    '''
    def un_normalize_data(self, new_data, parameter_index):
        means = np.array(self.normalization_factors[self.normalization_method]['mean'][parameter_index])
        std = np.array(self.normalization_factors[self.normalization_method]['std'][parameter_index])
        means = np.expand_dims(means, axis=1).repeat(new_data.shape[1], axis=1)
        std = np.expand_dims(std, axis=1).repeat(new_data.shape[1], axis=1)
        if self.normalization_method == 'zscore':
            return new_data * std + means
        else:
            return np.exp(new_data) * means





    '''
    Returns data of shape N x num_organisms x slice_size. Selects N random
    examples from all possible training samples. It selects from them evenly
    at the moment, but this can be tweaked to select more often from larger
    samples.
    '''
    def get_N_samples_and_targets(self, N, slice_size, train=True):
        samples = []  # Samples to feed to LSTM
        targets = []  # Single target to predict
        # gmeans = []  # Geometric means for weighting.
        if self.train_data is None:
            raise AttributeError('Please specify train and val data before '
                                 'calling this function.')

        # Samples from data based on number of samples. Ie, samples with more
        # data points get more selection.
        rands = np.random.rand(N)
        all_sum = np.sum([df.shape[1] for df in self.train_data])
        bins = np.cumsum([df.shape[1] / all_sum for df in self.train_data])
        which_samples = np.argmin(np.abs([bins - rand for rand in rands]),
                                  axis=1)

        # Pick a random sample and whether or not it is training or validation.
        for ws in which_samples:
            if train:
                sample = self.train_data[ws]
            else:
                sample = self.val_data[ws]
            # Pick a random starting point in the example. Get the data in
            # that slice and then the values immediately after.
            start_index = np.random.randint(sample.shape[1] - slice_size)
            data = sample.iloc[:, start_index: start_index + slice_size].values

            target = sample.iloc[:, start_index + slice_size].values
            # Store all the values
            samples.append(data)
            targets.append(target)

        samples = np.array(samples)
        targets = np.array(targets)
        return samples, targets


    def plot_values(self, sample_indices, num_strains):
        pass
