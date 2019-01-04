import sys
from biom import load_table
import numpy as np
import pandas as pd
import os
import argparse
'''
This is really just a helper script for performing some qiime functions on a
large number of files.

The structure of the files should be:
<parent dir>
    <study 1>
        taxonomy.fa
        biom.biom
        metadata.txt
    <study 2>
        taxonomy.fa
        ...
Each sub-directory should have the same number of files with the same file
extensions.
'''

# Read in our data
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="The directory full of directories.")
parser.add_argument("-t", "--taxonomy", type=str,
                    help="The file of green genes data.")

parser.add_argument("-r", "--reference", type=str,
                    help="The reference file of green genes data.")


args = parser.parse_args()

indir = args.input
tax_file = args.taxonomy
tax_reference = args.reference

metadata_string_base = 'biom add-metadata -i {} -o {} --sample-metadata-fp {}'
taxonomy_string_base = 'assign_taxonomy.py -i {} -r {} -t {} -o {}'

child_dirs = [os.path.join(indir, d) for d in os.listdir(indir)]
child_dirs = [d for d in child_dirs if os.path.isdir(d)]

for cd in child_dirs:
    all_files = os.listdir(cd)
    biom = [af for af in all_files if af.endswith('.biom')][0]
    fasta = [af for af in all_files if af.endswith('.fa')][0]
    metadata = [af for af in all_files if af.endswith('.txt')][0]
    metadata_to_execute = metadata_string_base.format(os.path.join(cd, biom),
                                                      os.path.join(cd, 'metadata_' + biom),
                                                      os.path.join(cd, metadata))
    taxonomy_to_execute = taxonomy_string_base.format(os.path.join(cd, fasta),
                                                      tax_reference,
                                                      tax_file,
                                                      os.path.join(cd, 'tax_output'))
    print(metadata_to_execute)
    print('\n')
    print(taxonomy_to_execute)
    print('\n------\n')
    os.system(metadata_to_execute)
    os.system(taxonomy_to_execute)
