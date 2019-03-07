import copy
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean, zscore

'''
This  is the main object that handles the OTU data.
It has a few main functions:
    1. Managing data normalization.
    2. Managing training, validation, and test data.
    3. Data sampling.
All of the different models use this data handler to get data samples.
'''

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
        # Read in the sample data.
        self.samples = []
        for f in files:
            self.samples.append(pd.read_csv(f, index_col=0))
        # If test_files are provided then read them in.
        if test_files is not None:
            self.test_data = []
            for f in test_files:
                self.test_data.append(pd.read_csv(f, index_col=0))
        else:
            self.test_data = None

        # Helper variables that are used elsewhere in the code.
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
        # For keeping track of the minimum size that the train/val data need to be.
        self.min_len = minsize
        # To be populated.
        self.train_data = []
        self.val_data = []
        temp_sizes = []
        # Go through each of the samples.
        for i, sample in enumerate(self.samples):
            # Where in the sample to set the split.
            index = int(percent * sample.shape[1])
            # If not enough examples, skip this file.
            if not ((sample.iloc[:, :index].shape[1]) < minsize or
                    (sample.iloc[:, index:].shape[1]) < minsize):
                # Append the new data to the appropriate location.
                self.train_data.append(sample.iloc[:, :index])
                self.val_data.append(sample.iloc[:, index:])
            else:
                # If it's too small only use it as training data.
                print('Skipping file, {}, because it\'s of shape: {}'.format(i, sample.shape))
                self.train_data.append(sample)

    def normalize_data(self, method='zscore'):
        '''
        Method for normalizing the input data. This can currently zscore and
        clr the data. Other methods could be added.
        '''
        method = method.lower()
        if method not in ['zscore', 'clr']:
            raise ValueError('Specify "zscore" or "clr" for method')
        if method == 'zscore':
            m = zscore
        else:
            m = clr

        # Keep track of the new data.
        new_vals = []
        for i, which_data in enumerate([self.samples, self.test_data]):
            # This is in case the test data is not supplied.
            if which_data is not None:
                new_vals = []
                # For each of the dataframes.
                for s in which_data:
                    # Apply the normalization method.
                    new_vals.append(pd.DataFrame(m(s.values, axis=0),
                                                 index=s.index,
                                                 columns=s.columns))

                # This is sort of a hack which is annoying.
                # Basically set the data to be the normalized version.
                if i == 0:
                    self.samples = new_vals
                elif i == 1:
                    self.test_data = new_vals

    def get_N_samples_and_targets(self, N, input_slice_size,
                                  target_slice_size,
                                  which_data='train',
                                  target_slice=True,
                                  slice_offset=1,
                                  which_donor=None):
        '''
        This is the main function for generating samples of data with which
        to train the neural network.

        Returns two things, an input and a target (both numpy arrays):
            input:[N x num_otus x input_slice_size]
            target: [N x num_otus x target_slice_size]

        Selects N random examples from all possible training samples.
        It selects from them based upon the number of timepoints present in
        each donor. So donors with more data get sampled more often.
        '''

        # What data source to sample from.
        which_data = which_data.lower()
        if which_data not in ['train', 'validation', 'test']:
            raise ValueError('Please specify either: train, validaiton, or test'
                             ' for argument "which_sample"')
        if self.test_data is None and which_data == 'test':
            raise ValueError('Do not select for test data when none is set.')
        # Set the appropriate location.
        if which_data == 'train':
            data_source = self.train_data
        elif which_data == 'validation':
            data_source = self.val_data
        elif which_data == 'test':
            data_source = self.test_data

        samples = []  # Samples to feed to the model.
        targets = []  # Targets to compare predictions against.

        # If train validation split hasn't been specified.
        if self.train_data is None:
            raise AttributeError('Please specify train and val data before '
                                 'calling this function.')
        # This flag is for generating samples from a particular donor.
        # During training this is always set as None so that it samples from
        # a distribution.
        if which_donor is None:
            # Samples from data based on number of samples. Ie, samples with more
            # data points get more selection.
            all_sizes = [d.shape[1] for d in data_source]
            probs = [s / (1.0 * sum(all_sizes)) for s in all_sizes]
            # Get the samples based on sizes.
            which_samples = np.random.choice(len(data_source), N, p=probs)
        else:
            # If we know what donor we want then only sample from that one.
            which_samples = [which_donor] * N

        # For each of the specified samples to pick.
        for ws in which_samples:
            sample = data_source[ws]

            # Pick a random starting point in the example. Get the data in
            # that slice and then the values immediately after.
            start_index = np.random.randint(sample.shape[1] - input_slice_size - target_slice_size)
            data = sample.iloc[:, start_index: start_index + input_slice_size].values

            # For the LSTM and EncoderDecoer we want a whole slice of values
            # to compare against. Not just a single target like in the FFN.
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
