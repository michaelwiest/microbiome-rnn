import pandas as pd
import numpy as np

class OTUHandler(object):
    def __init__(self, files):
        self.samples = []
        for f in files:
            self.samples.append(pd.read_csv(f, index_col=0))

        self.strains = list(self.samples[0].index.values)
        self.train_data = None
        self.val_data = None

    def set_train_val(self, percent=0.8):
        self.train_data = []
        self.val_data = []
        for sample in self.samples:
            index = int(percent * sample.shape[1])
            self.train_data.append(sample.iloc[:, :index])
            self.val_data.append(sample.iloc[:, index:])


    def get_N_samples_and_targets(self, N, slice_size, train=True):
        samples = []
        targets = []
        if self.train_data is None:
            raise AttributeError('Please specify train and val data before '
                                 'calling this function.')
        which_samples = np.random.randint(len(self.samples), size=N)
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
