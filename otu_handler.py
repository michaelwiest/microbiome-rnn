import copy
import pandas as pd
import numpy as np
# from skbio.stats.composition import clr, ilr
from scipy.stats.mstats import gmean, zscore


class OTUHandler(object):
    '''
    Class for handling OTU data. It generates samples and keeps track of
    training and validation data.
    '''
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


    def set_train_val(self, percent=0.8, minsize=12):
        '''
        Set the training and validation data for each sample. Can include
        a lower bound on size of train/validation
        '''
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


    def normalize_data(self, method='zscore'):
        '''
        Method for normalizing the input data. it also keeps track of the
        relevant parameters used for normalization so that they can be
        returned to their raw values for transformation of the predicted data.
        '''
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
                                                                    axis=0))
            if std is not None:
                self.normalization_factors[method]['std'].append(std(s.values,
                                                                     axis=0))
            new_vals.append(pd.DataFrame(m(s.values),
                                         index=s.index,
                                         columns=s.columns))

        self.samples = new_vals
        # Reassign the train and test values given the normalization.
        self.set_train_val()


    def un_normalize_data(self, new_data,
                          sample_index,
                          sample_timepoint_range):
        '''
        Function for returning the normalized values to the raw values.
        This is good for plotting the predicted values versus actual values.
        sample_timepoint_range: tuple or list of: (start_index, end_index)
        '''

        if (type(sample_timepoint_range) not in [list, tuple] or
            sample_timepoint_range[1] <= sample_timepoint_range[0]):
            raise ValueError('Please make the values fo the time range in '
                             'increasing order and a list or tuple.')


        means = np.array(self.normalization_factors[self.normalization_method]['mean'][sample_index])
        std = np.array(self.normalization_factors[self.normalization_method]['std'][sample_index])

        # Get the means and standard deviations of the input data over that
        # range and average it. These are used to "unnormalize" the input.
        means = np.mean(means[sample_timepoint_range[0]:
                              sample_timepoint_range[1]])
        std = np.mean(std[sample_timepoint_range[0]:
                          sample_timepoint_range[1]])
        if self.normalization_method == 'zscore':
            return new_data * std + means
        else:
            return np.exp(new_data) * means


    def get_N_samples_and_targets(self, N, slice_size,
                                  train=True, target_slice=True,
                                  slice_offset=1):
        '''
        Returns data of shape N x num_organisms x slice_size. Selects N random
        examples from all possible training samples. It selects from them evenly
        at the moment, but this can be tweaked to select more often from larger
        samples.
        '''
        samples = []  # Samples to feed to LSTM
        targets = []  # Single target to predict
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

            # For the LSTM we want a whole slice of values to compare against.
            # Not just a single target like in the FFN.
            # Now this just increments the input by one position for the target.
            if not target_slice:
                target = sample.iloc[:, start_index + slice_size].values
            else:
                target = sample.iloc[:, start_index + slice_offset:
                                        start_index + slice_size + slice_offset].values
            # Store all the values
            samples.append(data)
            targets.append(target)

        samples = np.array(samples)
        targets = np.array(targets)
        return samples, targets


    def plot_values(self, sample_indices, num_strains):
        pass
