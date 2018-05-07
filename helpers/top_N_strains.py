import pandas as pd
import sys
import numpy as np
from os import walk

from skbio.stats.composition import clr, ilr
from scipy.stats.mstats import gmean
from sklearn import cluster, covariance, manifold, decomposition, preprocessing
import os

'''
Takes the most prolific N strains from all the argument files.
Because there are instances where the taxonomy is repeated for a given file,
this takes the strain with the highest counts and uses that one.
'''


def load_data(file_to_read, subset=False, fraction_to_keep=0.05, index=None):
    otu_table = pd.read_csv(file_to_read, header=0, index_col=0)
    otu_table['mean'] = otu_table.mean(axis=1)
    otu_table.sort_values(by='mean', ascending=False, inplace=True)
    if subset:
        if index is None:
            otu_table = otu_table.head(int(otu_table.shape[0] * fraction_to_keep))
        else:
            print('Subsetting based on index.')
            otu_table = otu_table.loc[list(index)]
    otu_table = otu_table.drop('mean', axis=1)

    # Perform centered-log-ratio on the data to normalize it.
    otu_clr = otu_table.copy()
    clr_vals = clr(otu_clr.values)
    clr_vals = preprocessing.scale(clr_vals, axis=1)

    # Set values for clr data frame.
    for i in range(len(otu_table.columns)):
        otu_clr[otu_table.columns[i]] = clr_vals[:, i]

    return otu_table, otu_clr


def main():
    # Basic variables
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    N_strains = int(sys.argv[3])
    rolling_window = 25  # Configurable for smoothing

    # Get each file
    raws = []
    clrs = []
    files = []
    # Geometric means for transforming CLRs back to raw OTU counts
    gmeans = []
    for (dirpath, dirnames, filenames) in walk(input_dir):
        files.extend(filenames)
        break

    print(files)
    # Get the strains from each file.
    for f in files:
        raw, clr_i = load_data(os.path.join(input_dir, f), subset=False)
        raws.append(raw)
        clrs.append(clr_i)
        print('Finished loading: {}'.format(f))

    # This is what gets all of the appropriate strains from each file.
    # It uses set intersections to do this.
    strains = None
    i = 1
    while strains is None:
        indices = [set(otu.head(i).index.values) for otu in raws]
        intersection = set.intersection(*indices)

        if len(intersection) >= N_strains:
            strains = list(intersection)
        i += 1

    for i in range(len(clrs)):
        # Select only the rows with appropriate strains
        # These are all of the first occurences of each strain selected above.
        indices = [list(clrs[i].index == strain).index(True) for strain in strains]
        clrs[i] = clrs[i].iloc[indices]
        raws[i] = raws[i].iloc[indices]
        # Reindex so that they are all in the same order
        clrs[i] = clrs[i].reindex(strains)
        raws[i] = raws[i].reindex(strains)
        # Rename the columns so that they're just integers.
        clrs[i].columns = list(range(clrs[i].shape[1]))
        raws[i].columns = list(range(raws[i].shape[1]))
        # Take a rolling mean of the values (helps to smooth). Big improvements
        # in the model.
        if rolling_window is not None:
            clrs[i] = clrs[i].rolling(rolling_window, axis=1, min_periods=1).mean()
            raws[i] = raws[i].rolling(rolling_window, axis=1, min_periods=1).mean()

        # Write the values
        raw_fname = ''.join(files[i].split('.')[:-1]) + '_sub_{}.csv'.format(N_strains)
        clr_fname = ''.join(files[i].split('.')[:-1]) + '_sub_{}_clr.csv'.format(N_strains)

        raws[i].to_csv(os.path.join(output_dir, raw_fname))
        clrs[i].to_csv(os.path.join(output_dir, clr_fname))

        # gm = np.array(gmean(raws[i], axis=1))
        # gm = pd.DataFrame(np.expand_dims(gm, axis=0),
        #                   index=[raw_fname[:-4]],
        #                   columns=list(raws[i].index.values)
        #                   )
        # gmeans.append(gm)

    # pd.concat(gmeans).to_csv(os.path.join(gmeans_dir,
    #                                       '{}_gmeams.csv'.format(N_strains)
    #                                       ))
if __name__ == '__main__':
    main()
