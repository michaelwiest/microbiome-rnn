from DEICODE import untangle
import sys
import pandas as pd
import numpy as np

def main(fname):
    num_nonzero = 2
    df = pd.read_csv(fname, index_col=0)
    # Select rows with greater than 2 nonzero entries.
    good_indices = df.astype(bool).sum(axis=1).values > num_nonzero
    df = df.iloc[good_indices]
    # Normalize the counts
    meds = np.repeat(np.expand_dims(df.median(axis=1), 1), df.shape[1], axis=1)
    sums = np.repeat(np.expand_dims(df.sum(axis=1), 1), df.shape[1], axis=1)
    df = df * meds / sums
    completed = pd.DataFrame(untangle.complete_matrix(df.as_matrix().copy(),
                                                      iteration=100, minval=0.1),
                             index=df.index, columns=df.columns)

    out_name = ''.join(fname.split('.')[:-1]) + '_completed_normalized.csv'
    completed.to_csv(out_name)

if __name__ == '__main__':
    fname = sys.argv[1]
    main(fname)
