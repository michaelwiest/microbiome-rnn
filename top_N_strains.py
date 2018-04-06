import pandas as pd
import sys
import numpy as np
from skbio.stats.composition import clr, ilr
from sklearn import cluster, covariance, manifold, decomposition, preprocessing
import os

'''
Takes the most prolific N strains from the first argument file and then
selects those same strains from the second argument file. Writes them to
output directory as well. Also removes a specific slice of data from the
first file because the patient was travelling abroad at that time and his/her
gut was affected.
'''

def load_data(file_to_read, subset=False, fraction_to_keep=0.05, index=None):
    otu_table = pd.read_csv(file_to_read, header=0, index_col=0)

    # Plot the distribution of OTU counts.
    # means = np.sort(np.log(otu_table.mean(axis=1).values))

    otu_table = otu_table.transpose()
    # print(index)
    print('Subsetting without index.')
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
    clr_vals = clr(otu_table.values)
    clr_vals = preprocessing.scale(clr_vals, axis=1)

    # Set values for clr data frame.
    for i in range(len(otu_table.columns)):
        otu_clr[otu_table.columns[i]] = clr_vals[:, i]

    return otu_table, otu_clr

# The first file should be stool_A sample.
f1 = sys.argv[1]
f2 = sys.argv[2]
N_strains = int(sys.argv[3])
data_dir = 'data'
abroad = [71, 122]
rolling_window = 25

# Get the strains from each file.
otu_table1, otu_clr1 = load_data(f1)
otu_table2, otu_clr2 = load_data(f2)

strains = None
i = 1
while strains is None:
    i1 = set(otu_table1.head(i).index.values)
    i2 = set(otu_table2.head(i).index.values)
    intersection = set.intersection(i1, i2)
    if len(intersection) >= N_strains:
        strains = list(intersection)
    i += 1

# Get only the rows with the appropriate strains
otu_table1 = otu_table1.loc[strains]
otu_table2 = otu_table2.loc[strains]
otu_clr1 = otu_clr1.loc[strains]
otu_clr2 = otu_clr2.loc[strains]

# Reorder them rows so they're in the same order.
otu_table2 = otu_table2.reindex(strains)
otu_clr2 = otu_clr2.reindex(strains)
otu_table1 = otu_table1.reindex(strains)
otu_clr1 = otu_clr1.reindex(strains)


# Drop these because the patient was abroad and they probably skew the data.
# Also reindex the columns
otu_table1.drop(list(range(abroad[0], abroad[1])), axis=1)
otu_clr1.drop(list(range(abroad[0], abroad[1])), axis=1)
otu_table1.columns = list(range(otu_table1.shape[1]))
otu_clr1.columns = list(range(otu_clr1.shape[1]))



if otu_table1.shape[0] != otu_table2.shape[0]:
    raise ValueError('Not all strains specified in first argument file '
                     'are in second argument file.')

# Smooth out the data with rolling average.
if rolling_window is not None:
    otu_table1 = otu_table1.rolling(rolling_window, axis=1, min_periods=1).mean()
    otu_table2 = otu_table2.rolling(rolling_window, axis=1, min_periods=1).mean()
    otu_clr1 = otu_clr1.rolling(rolling_window, axis=1, min_periods=1).mean()
    otu_clr2 = otu_clr2.rolling(rolling_window, axis=1, min_periods=1).mean()

# Write the files.
otu_table1.to_csv(os.path.join(data_dir, 'gut_A_subset_{}.csv'.format(N_strains)))
otu_clr1.to_csv(os.path.join(data_dir, 'gut_A_subset_{}_clr.csv'.format(N_strains)))

otu_table2.to_csv(os.path.join(data_dir, 'gut_B_subset_{}.csv'.format(N_strains)))
otu_clr2.to_csv(os.path.join(data_dir, 'gut_B_subset_{}_clr.csv'.format(N_strains)))
