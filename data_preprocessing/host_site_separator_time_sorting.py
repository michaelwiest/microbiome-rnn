from biom import load_table
import numpy as np
import pandas as pd
import os
import argparse

'''
This file does the following:
- breaks out the biom tables into subjects and
  collection sites (stool, saliva, etc.).
- Adds taxonomy information to the files.
- Sorts the tables by collection date.
'''


def get_collection_days(table):
    '''
    This function is because one of the studies does not use a timestamp
    but uses a collection day which are ints stored as strings. of course...
    '''
    str_vals = [m['collection_day'] for m in table.metadata()]
    int_vals = [int(s) if s != '' else 0 for s in str_vals]
    return int_vals

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts. This file only
    takes one biom file at a time becuase there is a chance of naming
    collisions with doing multiple files.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def main():
    # Read in our arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--biom", type=str,
                        help="The BIOM file to handle.")
    parser.add_argument("-t", "--taxonomy", type=str,
                        help="The file or directory of taxonomy data.")
    parser.add_argument("-f", "--field", type=str, default='collection_timestamp',
                        help="The field to look in for timestamp info.")


    args = parser.parse_args()
    biom_name = args.biom
    biom_base = ''.join(biom_name.split('.')[:-1])
    tax_name = args.taxonomy
    field = args.field
    if field not in ['collection_timestamp', 'collection_day']:
        raise ValueError('field must be in [collection_timestamp, collection_day]')

    '''
    Load in the taxonomy data.
    '''
    if os.path.isdir(tax_name):
        tax_files = [os.path.join(tax_name, f) for f in os.listdir(tax_name) if f.endswith('.txt')]
        tax_dicts = []
        # Read in the files.
        for file in tax_files:
            tax_file = np.loadtxt(file, delimiter='\t', dtype=str)
            tax_dicts.append(dict(zip(tax_file[:, 0], tax_file[:, 1])))
        # Combine the dictionaries.
        mapping = merge_dicts(*tax_dicts)

    elif os.path.isfile(tax_name):
        tax_file = np.loadtxt(tax_name, delimiter='\t', dtype=str)
        mapping = dict(zip(tax_file[:, 0], tax_file[:, 1]))
    else:
        raise ValueError('Please check the file or directory being supplied'
                         'for the taxonomy.')
    print('Finished loading taxonomy')
    '''
    Break out into each sample based on the metadata.
    '''
    table = load_table(biom_name)
    output_tables = []
    output_fnames = []
    all_subjects = list(set([m['host_subject_id'] for m in table.metadata()]))
    all_samples = list(set([m['sample_type'] for m in table.metadata()]))
    all_subjects = [a for a in all_subjects if not a.lower().startswith('blank')]
    print('Subjects:\n{}'.format(all_subjects))
    print('Samples:\n{}'.format(all_samples))

    # Subset each of the files.
    for i, subject in enumerate(all_subjects):
        subject_fxn = lambda val, id_, md: md['host_subject_id'] == '{}'.format(subject)
        subject_sub = table.filter(subject_fxn, inplace=False)
        for sample in all_samples:
            sample_fxn = lambda val, id_, md: md['sample_type'] == '{}'.format(sample)
            sample_sub = subject_sub.filter(sample_fxn, inplace=False)
            # If its non-empty then add it to our output
            if sample_sub.shape[1] > 0:
                output_tables.append(sample_sub)
                new_name = biom_base + '_{}_{}'.format(subject, sample)
                output_fnames.append(new_name)
        print('Finished {} of {} subjects'.format(i + 1, len(all_subjects)))
    print(output_tables)
    print(output_fnames)

    '''
    Add the taxonomy and sort
    '''
    for j, table in enumerate(output_tables):

        # need this check for some reason because if not then it errors converting
        # to df.
        if table.shape[1] > 1:
            df = pd.DataFrame(table.to_dataframe())
            tv = df.values
            tcols = df.columns
            # Add taxonomy to each sample.
            indices = list(df.index.values)
            new_index = [mapping[i] for i in indices]
            to_save = pd.DataFrame(tv, index=new_index, columns=tcols)
            # Transpose because we will store the dates as a column and sort.
            to_save = to_save.T

            # If we want to sort the dates. Some samples don't have correct date
            # information associated so this doesn't work.
            if field == 'collection_timestamp':
                dates = [m['collection_timestamp'] for m in table.metadata()]
                to_save['date'] = pd.to_datetime(dates, infer_datetime_format=True,
                                                 errors='coerce')
            else:
                # For those bad samples that don't have time timestamps use
                # this method.
                dates = get_collection_days(table)
                print(dates)
                to_save['date'] = dates

            # Drop values with no timestamp.
            to_save.dropna(subset=['date'], inplace=True)
            # Sort by date.
            to_save.sort_values(by=['date'], inplace=True)
            # Capture the values to use as columns.
            dates = to_save['date']
            # Get rid of the column now that the values are sorted.
            to_save.drop(['date'], axis=1, inplace=True)
            # Transpose back to original shape.
            to_save = to_save.T
            # Reassign the time values as columns.
            to_save.columns = dates

            # Print relevant data and save the file.
            print(to_save.shape)
            output_fname = output_fnames[j] + '_sorted_tax.csv'
            print(output_fname)
            to_save.to_csv(output_fname)

if __name__ == '__main__':
    main()
