import pandas as pd
import numpy as np

# rerank batch based on some metric
def rerank_single(selected, metric, target):
    mbind = np.argmax(selected[metric])
    return selected[target][mbind]

# rerank single but ignore last thing in each batch
def rerank_nogold(selected, metric, target):
    mbind = np.argmax(selected[metric][:-1])
    return selected[target][mbind]

def rerank_df(df, scofunct, scoparam):
    reflist = list(df['ref'].unique())
    rrdf = []
    for r in reflist:
        # extract dataframe corresponding to smth
        exsamps = df[df['ref']==r]
        rrdf.append(scofunct(exsamps, scoparam[0], scoparam[1]))
    
    return sum(rrdf)/len(rrdf)