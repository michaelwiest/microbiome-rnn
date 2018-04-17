import pandas as pd
import numpy as np

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
    Set the training and validation data for each sample. Need to add
    a minimum threshold in here for having validation from a sample (ie, if
    sample is too small).
    '''
    def set_train_val(self, percent=0.8):
        self.train_data = []
        self.val_data = []
        temp_sizes = []
        for sample in self.samples:
            index = int(percent * sample.shape[1])
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
        samples = []
        targets = []
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

        for ws in which_samples:
            if train:
                sample = self.train_data[ws]
            else:
                sample = self.val_data[ws]
            start_index = np.random.randint(sample.shape[1] - slice_size)
            data = sample.iloc[:, start_index: start_index + slice_size].values
            target = sample.iloc[:, start_index + slice_size].values
            samples.append(data)
            targets.append(target)
        return np.array(samples), np.array(targets)
