'''
This file does the following:
- breaks out the biom tables into subjects and
  collection sites (stool, saliva, etc.).
- Adds taxonomy information to the files.
- Sorts the tables by collection date.
'''
import sys
from biom import load_table
import numpy as np
import pandas as pd

biom_name = sys.argv[1]
biom_base = ''.join(biom_name.split('.')[:-1])
tax_name = sys.argv[2]
sort_dates = True


'''
Break out into each sample based on the metadata.
'''
table = load_table(biom_name)
output_tables = []
output_fnames = []
all_subjects = list(set([m['host_subject_id'] for m in table.metadata()]))
all_samples = list(set([m['sample_type'] for m in table.metadata()]))

# Subset each of the files.
for subject in all_subjects:
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

print(output_tables)
print(output_fnames)

'''
Add the taxonomy and sort
'''
tax = np.loadtxt(tax_name, delimiter='\t', dtype=str)
mapping = dict(zip(tax[:, 0], tax[:, 1]))


for i, table in enumerate(output_tables):
    # Add taxonomy to each sample.
    df = pd.DataFrame(table.to_dataframe())
    tv = df.values
    tcols = df.columns
    indices = list(df.index.values)
    new_index = [mapping[i] for i in indices]
    to_save = pd.DataFrame(tv, index=new_index, columns=tcols)

    # If we want to sort the dates. Some samples don't have correct date
    # information associated so this doesn't work.
    if sort_dates:
        dates = [m['collection_timestamp'] for m in table.metadata()]
        to_save = to_save.T
        to_save['date'] = dates
        to_save.sort_values(by=['date'], inplace=True)
        to_save.drop(['date'], axis=1, inplace=True)
        to_save = to_save.T
    print(to_save.shape)
    to_save.to_csv(output_fnames[i] + '_sorted_tax.csv')
