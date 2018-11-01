import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def complete_and_multiindex_df(df):
    default_tax = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    s = list(df.index.values)
    lt = [list(ls.split(';')) for ls in s]
    for l in lt:
        if len(l) < len(default_tax):
            l += default_tax[-(len(default_tax)-len(l)):]

    ltn = pd.DataFrame(np.array(lt))
    ltn.index = df.index
    ltn.columns = default_tax
    combined = pd.concat((df, ltn), axis=1)
    combined.set_index(default_tax, inplace=True)
    return combined

def main():
    levels = 3
    df = pd.read_csv(sys.argv[1], index_col=0)
    combined = complete_and_multiindex_df(df)
    counts = combined.groupby(level=list(range(levels))).mean()

    cv = list(counts.values[:, 0])
    labels = ['\n'.join(c) for c in counts.index]
    fig, ax = plt.subplots(figsize=(15, 7))
    bar_width = 0.45
    rects1 = ax.bar(np.arange(len(cv)), cv, bar_width)
    ax.set_xticks(np.arange(len(cv)) + bar_width / 2)
    ax.set_xticklabels(labels,
                       ha='right',
                       fontsize=6)
    plt.ylabel('Strain Count')
    plt.title('Average Count of Strain Taxonomies')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
