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
    means = np.sort(np.log(otu_table.mean(axis=1).values))
    ind = np.arange(len(means))
    width = 0.35

    otu_table = otu_table.transpose()
    # print(index)
    if subset:
        if index is None:
            print('Subsetting without index.')
            otu_table['mean'] = otu_table.mean(axis=1)
            otu_table.sort_values(by='mean', ascending=False, inplace=True)
            otu_table = otu_table.head(int(otu_table.shape[0] * fraction_to_keep))
            otu_table = otu_table.drop('mean', axis=1)
        else:
            print('Subsetting based on index.')
            otu_table = otu_table.loc[list(index)]
    # Perform centered-log-ratio on the data to normalize it.
    otu_clr = otu_table.copy()
    clr_vals = clr(otu_table.values)
    clr_vals = preprocessing.scale(clr_vals, axis=1)
    # Set values for clr data frame.
    for i in range(len(otu_table.columns)):
        otu_clr[otu_table.columns[i]] = clr_vals[:, i]

    return otu_table, otu_clr


f1 = sys.argv[1]
f2 = sys.argv[2]
N_strains = int(sys.argv[3])
data_dir = 'data'
abroad = [71, 122]

otu_table1, otu_clr1 = load_data(f1, subset=True)
otu_table1 = otu_table1.head(N_strains)
otu_clr1 = otu_clr1.head(N_strains)
strains = list(otu_table1.index.values)
otu_table2, otu_clr2 = load_data(f2, subset=True, index=strains)
otu_table2 = otu_table2.reindex(strains)
otu_clr2 = otu_clr2.reindex(strains)


# Drop these because the patient was abroad and they probably skew the data.
# Also reindex the columns
otu_table1.drop(list(range(abroad[0], abroad[1])), axis=1)
otu_clr1.drop(list(range(abroad[0], abroad[1])), axis=1)
otu_table1.columns = list(range(otu_table1.shape[1]))
otu_clr1.columns = list(range(otu_clr1.shape[1]))

if otu_table1.shape[0] != otu_table2.shape[0]:
    raise ValueError('Not all strains specified in first argument file '
                     'are in second argument file.')

otu_table1.to_csv(os.path.join(data_dir, 'gut_A_subset_{}.csv'.format(N_strains)))
otu_clr1.to_csv(os.path.join(data_dir, 'gut_A_subset_{}_clr.csv'.format(N_strains)))

otu_table2.to_csv(os.path.join(data_dir, 'gut_B_subset_{}.csv'.format(N_strains)))
otu_clr2.to_csv(os.path.join(data_dir, 'gut_B_subset_{}_clr.csv'.format(N_strains)))
