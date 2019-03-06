import pandas as pd
import sys
import numpy as np
from os import walk
import argparse
import os

'''
Takes the most prolific N OTUs from all the files in the argument directory.
If there are fewer than N OTUs that intersect all files then the maximum
number of intersecting strains is used instead.
Usage:

python top_N_strains.py -i <input dir> -o <output dir>  -n <number of otus>
'''

def main():
    # Read in our arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="The directory of input data.")
    parser.add_argument("-o", "--output", type=str,
                        help="The directory of output data.")
    parser.add_argument("-n", "--notus", type=int, default=100,
                        help="How many OTUs to subset to.")

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    N_strains = args.notus

    # DEPRECATED. This could be reimplemented but I did not.
    rolling_window = None  # Configurable for smoothing

    # Get all the filenames
    files = []
    for (dirpath, dirnames, filenames) in walk(input_dir):
        files.extend(filenames)
        break
    files = [f for f in files if f.endswith('.csv')]
    files.sort()
    print('Loading...')
    print('\n'.join(files))

    # Read in the data.
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
