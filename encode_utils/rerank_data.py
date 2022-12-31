import numpy as np
import random

# rerank batch based on some metric
def rerank_single(selected, metric, target):
    if metric=="posthoc":
        mbind = np.argmin(selected[metric])
    else:
        mbind = np.argmax(selected[metric])
    return list(selected[target])[mbind]

# rerank single but ignore last thing in each batch
def rerank_nogold(selected, metric, target):
    if metric=="posthoc":
        mbind = np.argmin(list(selected[metric])[:-1])
    else:
        mbind = np.argmax(list(selected[metric])[:-1])
    return list(selected[target])[mbind]

# rerank single but ignore last thing in each batch
def rerank_nogweight(selected, metric, target):
    met = list(selected[metric])
    phoc = list(selected['posthoc'])
    newvec = [met[i]*.9-phoc[i]*1.25 for i in range(len(met))]
    mbind = np.argmax(newvec[:-1])
    return list(selected[target])[mbind]

# TODO changes for with / without gold
def rerank_weighted(selected, metric, target):
    met = list(selected[metric])
    phoc = list(selected['posthoc'])
    newvec = [met[i]*.9-phoc[i]*1.25 for i in range(len(met))]
    mbind = np.argmax(newvec)
    return list(selected[target])[mbind]

def rerank_rand(selected, metric, target):
    return random.choice(list(selected[target])[:-1])

def rerank_df(df, scofunct, scoparam):
    reflist = list(df['ref'].unique())
    rrdf = []
    for r in reflist:
        if type(r) is not str:
            continue
        if len(r)<5:
            continue
        # extract dataframe corresponding to smth
        exsamps = df[df['ref']==r]
        rrdf.append(scofunct(exsamps, scoparam[0], scoparam[1]))
    return sum(rrdf)/len(rrdf)

def rerank_dist(df, scofunct, scoparam):
    reflist = list(df['ref'].unique())
    rrdf = []
    for r in reflist:
        # extract dataframe corresponding to smth
        exsamps = df[df['ref']==r]
        rrdf.append(scofunct(exsamps, scoparam[0], scoparam[1]))
    return rrdf