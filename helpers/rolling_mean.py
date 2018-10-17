import pandas as pd
import sys
import os

input_dir = sys.argv[1]
output_dir = sys.argv[2]
window_size = int(sys.argv[3])

input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
csvs = [pd.read_csv(f) for f in input_files]

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for i, f in enumerate(input_files):
    new_df = csvs[i].rolling(window_size, axis=1, min_periods=1).mean()
    new_df.to_csv(os.path.join(output_dir, os.path.basename(f)))
