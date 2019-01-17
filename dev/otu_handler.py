import copy
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean, zscore

def clr(some_array, axis):
    '''
    Had to reimplement this because the server is python2 and
    skbio is only python3
    '''
    gms = gmean(some_array, axis=axis)
    gms = np.expand_dims(gms, axis=axis)
    gms = np.repeat(gms, some_array.shape[axis], axis=axis)
    return np.log(some_array / gms)



class OTUHandler(object):
    '''
    Class for handling OTU data. It generates samples and keeps track of
    training and validation data.
    '''
    def __init__(self, files, test_files=None):
        self.samples = []
        for f in files:
            self.samples.append(pd.read_csv(f, index_col=0))
        if test_files is not None:
            self.test_data = []
            for f in test_files:
                self.test_data.append(pd.read_csv(f, index_col=0))
        else:
            self.test_data = None

        # Keep track of these for getting back and forth from normalized.
        self.raw_samples = copy.deepcopy(self.samples)
        self.strains = list(self.samples[0].index.values)
        self.num_strains = len(self.strains)
        # These get set when calling set_train_val
        self.train_data = None
        self.val_data = None

    def set_train_val(self, percent=0.8, minsize=20):
        '''
        Set the training and validation data for each sample. Can include
        a lower bound on size of train/validation
        '''
        # For keeping track of max size the slice can be.
        self.min_len = minsize
        # To be populated.
        self.train_data = []
        self.val_data = []
        temp_sizes = []
        for i, sample in enumerate(self.samples):
            index = int(percent * sample.shape[1])
            # If not enough examples, skip this file.
            if not ((sample.iloc[:, :index].shape[1]) < minsize * 2 or
                    (sample.iloc[:, index:].shape[1]) < minsize * 2):
                self.train_data.append(sample.iloc[:, :index])
                self.val_data.append(sample.iloc[:, index:])
                temp_sizes.append(sample.iloc[:, :index].shape[1])
                temp_sizes.append(sample.iloc[:, index:].shape[1])

            else:
                print('Skipping file, {}, because it\'s of shape: {}'.format(i, sample.shape))
                self.train_data.append(sample)
                temp_sizes.append(sample.shape[1])

    def normalize_data(self, method='zscore'):
        '''
        Method for normalizing the input data.
        '''
        method = method.lower()
        if method not in ['zscore', 'clr']:
            raise ValueError('Specify "zscore" or "clr" for method')
        if method == 'zscore':
            m = zscore
        else:
            m = clr

        new_vals = []
        for i, which_data in enumerate([self.samples, self.test_data]):
            if which_data is not None:
                new_vals = []
                for s in which_data:
                    new_vals.append(pd.DataFrame(m(s.values, axis=0),
                                                 index=s.index,
                                                 columns=s.columns))
                # This is sort of a hack which is annoying.
                if i == 0:
                    self.samples = new_vals
                elif i == 1:
                    self.test_data = new_vals

    def get_N_samples_and_targets(self, N, input_slice_size,
                                  target_slice_size,
                                  which_data='train',
                                  target_slice=True,
                                  slice_offset=1):
        '''
        Returns data of shape N x num_organisms x input_slice_size. Selects N random
        examples from all possible training samples. It selects from them evenly
        at the moment, but this can be tweaked to select more often from larger
        samples.
        '''
        which_data = which_data.lower()
        if which_data not in ['train', 'validation', 'test']:
            raise ValueError('Please specify either: train, validaiton, or test'
                             ' for argument "which_sample"')
        if self.test_data is None and which_data == 'test':
            raise ValueError('Do not select for test data when none is set.')

        if which_data == 'train':
            data_source = self.train_data
        elif which_data == 'validation':
            data_source = self.val_data
        elif which_data == 'test':
            data_source = self.test_data

        samples = []  # Samples to feed to LSTM
        targets = []  # Single target to predict
        if self.train_data is None:
            raise AttributeError('Please specify train and val data before '
                                 'calling this function.')

        # Samples from data based on number of samples. Ie, samples with more
        # data points get more selection.
        all_sizes = [d.shape[1] for d in data_source]
        probs = [s / (1.0 * sum(all_sizes)) for s in all_sizes]
        which_samples = np.random.choice(len(data_source), N,
                                         p=probs)

        # Pick a random sample and whether or not it is training or validation.
        for ws in which_samples:
            sample = data_source[ws]

            # Pick a random starting point in the example. Get the data in
            # that slice and then the values immediately after.
            start_index = np.random.randint(sample.shape[1] - input_slice_size - target_slice_size)
            data = sample.iloc[:, start_index: start_index + input_slice_size].values

            # For the LSTM we want a whole slice of values to compare against.
            # Not just a single target like in the FFN.
            # Now this just increments the input by one position for the target.
            if not target_slice:
                target = sample.iloc[:, start_index + input_slice_size].values
            else:
                target = sample.iloc[:, start_index + slice_offset:
                                        start_index + target_slice_size + slice_offset].values
            # Store all the values
            samples.append(data)
            targets.append(target)

        samples = np.array(samples)
        targets = np.array(targets)

        return samples, targets


    def plot_values(self, sample_indices, num_strains):
        pass
