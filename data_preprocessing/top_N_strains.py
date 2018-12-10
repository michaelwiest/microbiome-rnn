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
Takes the most prolific N strains from all the argument files.
Because there are instances where the taxonomy is repeated for a given file,
this takes the strain with the highest counts and uses that one.
'''

def get_string_index_from_multiindex_df(df, join_char=';'):
    levels = len(df.index.levels)
    str_lists = [df.index.get_level_values(level=l).values.tolist()
                 for l in list(range(levels))]
    zipped = list(zip(*str_lists))
    return [join_char.join(z) for z in zipped]



def complete_and_multiindex_df(df, split_char=';', depth=7):
    default_tax = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    s = list(df.index.values)
    lt = [list(ls.split(split_char)) for ls in s]
    lt = [[ltii.strip() for ltii in lti[:depth]] for lti in lt]
    for l in lt:
        if len(l) < depth:
            l += default_tax[-(len(default_tax)-len(l)):depth]
    ltn = pd.DataFrame(np.array(lt))
    ltn.index = df.index

    ltn.columns = default_tax[:depth]
    combined = pd.concat((df, ltn), axis=1)
    combined.set_index(default_tax[:depth], inplace=True)
    return combined

def load_data_and_sort(file_to_read):
    '''
    Need to sort the data here for the first() function call in groupby.
    Although that likely won't be used anymore.
    '''
    otu_table = pd.read_csv(file_to_read, header=0, index_col=0)
    otu_table['mean'] = otu_table.mean(axis=1)
    otu_table.sort_values(by='mean', ascending=False, inplace=True)

    otu_table = otu_table.drop('mean', axis=1)

    return otu_table


def main():
    # Read in our data
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="The directory of input data.")
    parser.add_argument("-o", "--output", type=str,
                        help="The directory of output data.")
    parser.add_argument("-n", "--nstrains", type=int, default=100,
                        help="How many strains to subset to.")
    parser.add_argument("-l", "--level", type=int,
                        help="Taxonomy level to clip to. Six is genus.",
                        default=6)
    parser.add_argument('-s', "--sum",
                        help="should the OTUs be summed",
                        action='store_true')
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    level = args.level
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    N_strains = args.nstrains
    sum_strains = args.sum

    # Not really used.
    rolling_window = None  # Configurable for smoothing

    # Get each file
    otu_dfs = []
    files = []
    for (dirpath, dirnames, filenames) in walk(input_dir):
        files.extend(filenames)
        break

    print('Loading...')
    print('\n'.join(files))

    # Get the OTUs from each file.
    for f in files:
        raw = load_data_and_sort(os.path.join(input_dir, f))
        otu_dfs.append(complete_and_multiindex_df(raw, depth=level))
    print('Finished loading data')

    for i, df in enumerate(otu_dfs):
        # Either take the first (highest) value for each strain. Or
        # get the sum of all the matching strains.
        if sum_strains:
            grouped = df.groupby(level=list(range(0, level))).sum()
            grouped['mean'] = grouped.mean(axis=1)
            grouped.sort_values(by='mean', inplace=True, ascending=False)
            grouped.drop('mean', axis=1, inplace=True)
        else:
            grouped = df.groupby(level=list(range(0, level))).first()
        # Reindex the df so that it has a string index.
        new_index = get_string_index_from_multiindex_df(grouped)
        grouped.index = new_index
        otu_dfs[i] = grouped


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

        # Rename the columns so that they're just integers.
        otu_dfs[i].columns = list(range(otu_dfs[i].shape[1]))
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
