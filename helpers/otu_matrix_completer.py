from DEICODE import untangle
import sys
import pandas as pd


def main():
    fname = sys.argv[1]
    df = pd.read_csv(fname, index_col=0)
    # df = df.T
    completed = pd.DataFrame(untangle.complete_matrix(df.as_matrix().copy(),
                                                      iteration=100, minval=0.1),
                             index=df.index, columns=df.columns)
    out_name = ''.join(fname.split('.')[:-1]) + '_completed.csv'
    completed.to_csv(out_name)

if __name__ == '__main__':
    main()