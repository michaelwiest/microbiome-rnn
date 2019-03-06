import os
import argparse

'''
This is really just a helper script for performing some QIIME
and BIOM functions on a large number of files.

While not explicitly listed as an import dependancy this script needs
a version of QIIME1 and BIOM to be installed. Specifically the commands for:
    add-metadata (BIOM)
    assign_taxonomy.py (QIIME1)

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
extensions. There should not be files other than those three types listed above
in each subdirectory.

This script does two things for each sub directory:
-Adds metadata to each study such as donor info, timestamps, etc.
-Generates a lookup between the sequencing fragment from the biom file and
 an assigned taxonomy.

Both of these are used later in the pipeline.
'''

# Read in our arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="The directory full of directories.")
parser.add_argument("-t", "--taxonomy", type=str,
                    help="The file of green genes data. "
                         "These are in the taxonomy directory.")

parser.add_argument("-r", "--reference", type=str,
                    help="The reference file of green genes data. "
                         "These are in the rep-set directory of green genes.")


args = parser.parse_args()

indir = args.input
tax_file = args.taxonomy
tax_reference = args.reference

# These are the command line commands that will be executed.
metadata_string_base = 'biom add-metadata -i {} -o {} --sample-metadata-fp {}'
taxonomy_string_base = 'assign_taxonomy.py -i {} -r {} -t {} -o {}'

# Get each of the child directories from the parent directory.
child_dirs = [os.path.join(indir, d) for d in os.listdir(indir)]
child_dirs = [d for d in child_dirs if os.path.isdir(d)]

for cd in child_dirs:
    all_files = os.listdir(cd)
    # This captures each of the relevant files in each subdir.
    # As mentioned above there should only be one with each file extension.
    # If there are more than one, then only the first will be taken.
    biom = [af for af in all_files if af.endswith('.biom')][0]
    fasta = [af for af in all_files if af.endswith('.fa')][0]
    metadata = [af for af in all_files if af.endswith('.txt')][0]

    # Add metadata to the biom file.
    metadata_to_execute = metadata_string_base.format(os.path.join(cd, biom),
                                                      os.path.join(cd, 'metadata_' + biom),
                                                      os.path.join(cd, metadata))
    # Get a taxonomy mapping.
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
