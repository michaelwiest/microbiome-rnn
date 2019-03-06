import pandas as pd
import sys
import numpy as np
from os import walk
import argparse
# from skbio.stats.composition import clr, ilr
# from scipy.stats.mstats import gmean
# from sklearn import cluster, covariance, manifold, decomposition, preprocessing
import os

'''
Takes the most prolific N strains from all the files in the argument directory.
'''

def main():
    # Read in our data
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="The directory of input data.")
    parser.add_argument("-o", "--output", type=str,
                        help="The directory of output data.")
    parser.add_argument("-n", "--nstrains", type=int, default=100,
                        help="How many strains to subset to.")

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    N_strains = args.nstrains

    # Not really used.
    rolling_window = None  # Configurable for smoothing

    # Get each file
    files = []
    for (dirpath, dirnames, filenames) in walk(input_dir):
        files.extend(filenames)
        break

    print('Loading...')
    print('\n'.join(files))

    # Get the OTUs from each file.

    otu_dfs = [pd.read_csv(os.path.join(input_dir, f), header=0, index_col=0) for f in files if f.endswith('.csv')]
    print('Finished loading data')

    # This is what gets all of the appropriate strains from each file.
    # It uses set intersections to do this.
    strains = None
    i = 1
    df_shapes = [r.shape[0] for r in otu_dfs]
    while i < max(df_shapes):
        indices = [set(otu.head(i).index.values) for otu in otu_dfs]
        intersection = set.intersection(*indices)
        strains = list(intersection)
        if len(strains) >= N_strains:
            break
        i += 1

    num_strains_matched = len(strains)
    print('Got intersection strains. There are {}'.format(num_strains_matched))

    for i in range(len(otu_dfs)):
        otu_dfs[i] = otu_dfs[i].reindex(strains)

        # Take a rolling mean of the values (helps to smooth). Big improvements
        # in the model.
        if rolling_window is not None:
            otu_dfs[i] = otu_dfs[i].rolling(rolling_window, axis=1, min_periods=1).mean()

        # Write the values
        raw_fname = ''.join(files[i].split('.')[:-1]) + '_sub_{}.csv'.format(min(N_strains,
                                                                                 num_strains_matched))

        otu_dfs[i].to_csv(os.path.join(output_dir, raw_fname))


if __name__ == '__main__':
    main()
