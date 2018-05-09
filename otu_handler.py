import pandas as pd
import numpy as np
# from skbio.stats.composition import clr, ilr
from scipy.stats.mstats import gmean

'''
Class for handling OTU data. It generates samples and keeps track of
training and validation data.
'''
class OTUHandler(object):
    def __init__(self, files):
        self.samples = []
        for f in files:
            self.samples.append(pd.read_csv(f, index_col=0))

        self.strains = list(self.samples[0].index.values)
        self.num_strains = len(self.strains)
        self.train_data = None
        self.val_data = None

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
    Returns data of shape N x num_organisms x slice_size. Selects N random
    examples from all possible training samples. It selects from them evenly
    at the moment, but this can be tweaked to select more often from larger
    samples.
    '''
    def get_N_samples_and_targets(self, N, slice_size, train=True):
        samples = []  # Samples to feed to LSTM
        targets = []  # Single target to predict
        gmeans = []  # Geometric means for weighting.
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

            # Get the geometric mean of the data sampled. This is for
            # a hacked CLR on the test examples.
            gm = gmean(data, axis=1)
            # data = clr(data) # Perform CLR on the train data.
            # This is a HACK that i'm using in absence of CLR from skbio.
            gmt = np.expand_dims(gm, 1).repeat(slice_size, 1)
            data = np.log(data / gmt)
            # Hacked CLR on the targets.
            target = np.log(sample.iloc[:, start_index + slice_size].values / gm)
            # Store all the values
            samples.append(data)
            targets.append(target)
            gmeans.append(gm)

        samples = np.array(samples)
        targets = np.array(targets)
        # Expand the dimensions of the gmeans to match that of the samples.
        axis_to_add = len(samples.shape) - 1
        gmeans = np.expand_dims(gmeans, axis_to_add)#.repeat(slice_size,
                                                    #        axis=axis_to_add)
        return samples, targets, gmeans
