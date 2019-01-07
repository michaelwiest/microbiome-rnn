import sys
from biom import load_table
import os
import argparse
import biom
import pdb

'''
This file combines multiple biom files from the (hopefully) same individuals
'''

# Read in our data
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="Directory of biom files to combine.")
parser.add_argument("-o", "--output", type=str,
                    help="Output biom file name.")
args = parser.parse_args()
indir = args.input
outname = args.output

biom_file_names = [os.path.join(indir, f) for f in os.listdir(indir)]
biom_files = [load_table(f) for f in biom_file_names]

# Combine them all together.
for i, bf in enumerate(biom_files):
    if i == 0:
        output = bf
    else:
        output = output.merge(bf)
    print('Completed {} of {}'.format(i + 1, len(biom_files)))
    print('Output shape is: {}'.format(output.shape))

# Write the output.
with open('{}'.format(outname), 'w') as f:
    f.write(output.to_json("example"))
