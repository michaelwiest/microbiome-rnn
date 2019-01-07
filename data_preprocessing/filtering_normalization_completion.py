from DEICODE import untangle
import sys
import pandas as pd
import numpy as np
import argparse
import os


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
    dfs = [pd.read_csv(f, index_col=0, header=0) for f in files]

    for i, df in enumerate(dfs):
        # Select rows with greater than the specified nonzero entries.
        good_indices = df.astype(bool).sum(axis=1).values > num_nonzero
        df = df.iloc[good_indices]

        # Normalize the counts
        sums = np.repeat(np.expand_dims(df.sum(axis=0), 0),
                         df.shape[0], axis=0)
        meds = np.median(sums)
        df = df * meds / sums

        # Perform matrix completion.
        completed = pd.DataFrame(untangle.complete_matrix(df.as_matrix().copy(),
                                                          iteration=100, minval=0.1),
                                 index=df.index, columns=df.columns)

        out_name = ''.join(files[i].split('.')[:-1]) + '_completed_normalized.csv'
        completed.to_csv(os.path.join(outdir, out_name))


if __name__ == '__main__':
    main()
