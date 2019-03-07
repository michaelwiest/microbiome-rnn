import pandas as pd
import sys
import numpy as np
from os import walk
import argparse
import os

'''
This file takes OTUs with the same name and sums their counts. This is to
get rid of redundant OTUs that are listed as different but are actually the
same.
'''


def get_string_index_from_multiindex_df(df, join_char=';'):
    '''
    Given a multiindexed dataframe this returns a string representation of
    those fields all joined by the join_char.
    '''
    levels = len(df.index.levels)
    # Get all of the index values.
    str_lists = [df.index.get_level_values(level=l).values.tolist()
                 for l in list(range(levels))]
    # Join them all back together.
    zipped = list(zip(*str_lists))
    return [join_char.join(z) for z in zipped]


def complete_and_multiindex_df(df, split_char=';', depth=7):
    '''
    This takes a taxonomic index separated by the split_char and
    multiindexes the dataframe so that groupby functions can be used.

    depth is how far into the taxonomic tree to keep the values. 0 = kingdom.
    6 = genus.
    '''
    # Default taxonomic levels. Often times these are missing.
    default_tax = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    # Current index values.
    s = list(df.index.values)
    # Split each entry of the index on the split char.
    # This results in a list of lists.
    split_taxonomy = [list(ls.split(split_char)) for ls in s]
    # Removes whitespace around the taxonomy data. This also gets rid of
    # tax data past the specified taxonomic depth.
    split_taxonomy = [[ltii.strip() for ltii in lti[:depth]] for lti in split_taxonomy]
    # For each taxonomic entry, i there are missing tax entries then fill them
    # in using the default_tax field.
    for tax_entry in split_taxonomy:
        if len(tax_entry) < depth:
            tax_entry += default_tax[-(len(default_tax)-len(tax_entry)):depth]

    # Make a dataframe using the taxonomic data.
    taxonomy_df = pd.DataFrame(np.array(split_taxonomy))
    # Set the index as the index of the input df.
    taxonomy_df.index = df.index
    # Set the columns to be the appropriate taxonomic labels.
    taxonomy_df.columns = default_tax[:depth]
    # Basically store the data from the input df into this new dataframe.
    combined = pd.concat((df, taxonomy_df), axis=1)
    # Now set the index to be the multiindex.
    combined.set_index(default_tax[:depth], inplace=True)
    return combined

def load_data_and_sort(file_to_read):
    '''
    Basically this loads in data and sorts the OTUs by their average value.
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
    parser.add_argument("-l", "--level", type=int,
                        help="Taxonomy level to clip to. Six is genus.",
                        default=6)
    parser.add_argument('-s', "--sum",
                        help="should the OTUs be summed",
                        action='store_true', default=True)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    level = args.level
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sum_strains = args.sum

    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    otu_dfs = []
    # Get the OTUs from each file.
    for f in files:
        raw = load_data_and_sort(os.path.join(input_dir, f))
        otu_dfs.append(complete_and_multiindex_df(raw, depth=level))
        print('Loaded: {}'.format(f))
    print('Finished loading data')

    for i, df in enumerate(otu_dfs):
        # Either take the first (highest) value for each strain. Or
        # get the sum of all the matching strains. I only use the sum function.
        if sum_strains:
            grouped = df.groupby(level=list(range(0, level))).sum()
            grouped['mean'] = grouped.mean(axis=1)
            grouped.sort_values(by='mean', inplace=True, ascending=False)
            grouped.drop('mean', axis=1, inplace=True)
        # This isn't really used.
        else:
            grouped = df.groupby(level=list(range(0, level))).first()
        # Reindex the df so that it has a string index.
        new_index = get_string_index_from_multiindex_df(grouped)
        grouped.index = new_index
        otu_dfs[i] = grouped

    # Write each of the files.
    for i, to_write in enumerate(otu_dfs):
        raw_fname = ''.join(files[i].split('.')[:-1]) + '_truncated_sorted.csv'
        to_write.to_csv(os.path.join(output_dir, raw_fname))


if __name__ == '__main__':
    main()
