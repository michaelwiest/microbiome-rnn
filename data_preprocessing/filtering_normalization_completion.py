from DEICODE import untangle
import sys
import pandas as pd
import numpy as np
import argparse
import os

'''
This file does three main things. 
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="The directory of input data.")
    parser.add_argument("-o", "--output", type=str,
                        help="The directory of output data.")
    parser.add_argument("-n", "--nonzero", type=int, default=10,
                        help="The number of nonzero entrires necessary.")

    args = parser.parse_args()
    num_nonzero = args.nonzero
    indir = args.input
    outdir = args.output

    # Make the output directory
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # Read in all the files.
    files = os.listdir(indir)
    files = [f for f in files if f.endswith('.csv')]
    files.sort()

    dfs = [pd.read_csv(os.path.join(indir, f), index_col=0, header=0)
           for f in files if f.endswith('.csv')]
    print('Finished loading data.')
    for i, df in enumerate(dfs):
        out_name = ''.join(files[i].split('.')[:-1]) + '_completed_normalized.csv'
        print(out_name)
        print('input shape: {}'.format(df.shape))
        # Select rows with greater than the specified nonzero entries.
        good_indices = df.astype(bool).sum(axis=1).values >= num_nonzero
        df = df.iloc[good_indices]
        # This removes timepoints with all zero counts
        df = df.loc[:, (df != 0).any(axis=0)]
        print('output shape: {}'.format(df.shape))
        # Normalize the counts
        sums = np.repeat(np.expand_dims(df.sum(axis=0), 0),
                         df.shape[0], axis=0)
        meds = np.median(sums)
        df = df * meds / sums
        print('normalized')
        # Perform matrix completion.
        try:
            completed = pd.DataFrame(untangle.complete_matrix(df.as_matrix().copy(),
                                                              iteration=1000, minval=0.1),
                                                              index=df.index, columns=df.columns)
            print('completed')
            completed.to_csv(os.path.join(outdir, out_name))
        except:
            print('Failed to complete this file.')



if __name__ == '__main__':
    main()
