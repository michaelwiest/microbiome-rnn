import sys
from biom import load_table
import numpy as np
import pandas as pd

biom_name = sys.argv[1]
tax_name = sys.argv[2]

table = pd.DataFrame(load_table(biom_name).to_dataframe())
tv = table.values
tcols = table.columns
tax = np.loadtxt(tax_name, delimiter='\t', dtype=str)
print('Finished loading data.')

indices = list(table.index.values)
mapping = dict(zip(tax[:, 0], tax[:, 1]))
new_index = [mapping[i] for i in indices]
to_save = pd.DataFrame(tv, index=new_index, columns=tcols)
output_name = ''.join(biom_name.split('.')[:-1]) + '_tax_headers.csv'
to_save.to_csv(output_name)
