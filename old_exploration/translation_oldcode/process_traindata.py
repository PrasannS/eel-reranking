import json
import random
from mt_scores import get_scores_auto
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from os.path import exists
import pickle


#random.seed(1)

# load in data from a candidates file as generated in earlier code
def load_cands(fname):
    data = []
    with open(fname, 'r') as file:
        while True:
            line = file.readline()
            # if line is empty
            # end of file is reached
            if not line or len(line)<3:
                break

            data.append(json.loads(line))
    return data

def cands_df(fname):
    cands = load_cands("candoutputs/"+fname)
    result = []
    for c in cands:
        for i in range(len(c['scores'])):
            result.append({
                "adhoc":c['scores'][i],
                "ahyp":c['cands'][i],
                "src": c['inp'],
                "ref": c["ref"], 
            })
    return pd.DataFrame(result)
    
# extract sized batches from each lattice, populate df with data
def lset_tocominp(lset, batchsize, exsamps):
    res = []
    for g in lset:
        if len(g['cands'])<batchsize:
            continue
        # repeat process exsamps number of times
        for i in range(exsamps):
            samp = random.sample(g['cands'], batchsize-1)+[g['ref']]
            for s in samp:
                tmp = {}
                tmp['src'] = g['inp']
                tmp['hyp'] = s
                tmp['ref'] = g['ref']
                res.append(tmp)
    return pd.DataFrame(res)

# tokenizes inputs for mt data, gives posadd for masking purposes
MAXTOKS = 512
def tokenize_inputs(inpdf, toker):
    xinp = []
    yinp = []
    padd_inp = []
    print("tokenizing input")
    for index, row in inpdf.iterrows():
        if index%10000==0:
            print(index)
        #print(row['c1'], row['c2'])
        # will need to make a custom mask (maybe) so that inputs from both languages are encoded separately
        toktmp = toker(row['src']).input_ids
        lent = len(toktmp)
        hyptmp = toker(row['hyp']).input_ids
        toktmp.extend(hyptmp)
        toktmp = toktmp[:min(len(toktmp), MAXTOKS)]
        padd_inp.append(lent)
        xinp.append(toktmp)
        yinp.append(row['score'])
    return xinp, yinp, padd_inp

# get dataframe with inputs for exploded, can then score for later
# reference
def lset_to_explodeddf(lset):
    res = []
    for g in lset:
        samp = g['cands']
        for s in samp:
            tmp = {}
            tmp['src'] = g['inp']
            tmp['hyp'] = s
            tmp['ref'] = g['ref']
            res.append(tmp)
    return pd.DataFrame(res)

exploded_dirs = {
    "en_de":"./candoutputs/explodedmt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.904_0.0_0.9.jsonl",
    "fr_en":"./candoutputs/explodedmtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.904_0.0_0.9.jsonl",
    "en_deb1":"./candoutputs/testbeam1en_de.jsonl",
    "en_deb50":"./candoutputs/testbeam50en_de.jsonl"
}

# let's say we want to get 4 batches from 32 (including ref, which is gold)
def get_bin_sample(subdf):
    #print(subdf)
    bins = pd.qcut(subdf['score'], 4, precision=1)
    ref = subdf.iloc[-1]
    subdf['qbin'] = bins
    ubins = bins.unique()
    res = []
    for i in range(0, len(ubins)-1):
        tmp = subdf[subdf['qbin']==ubins[i]].sample(replace=True)
        res.append(tmp.iloc[-1])
    res.append(ref)
    return pd.DataFrame(res).drop(columns=['Unnamed: 0'])

def convert_bin_sample(fulldf, exsamps):
    assert len(fulldf)%32==0
    
    print(len(fulldf)/32)
    newdf = pd.DataFrame()
    fullres = []
    for i in range(int(len(fulldf)/32)):
        for j in range(exsamps):
            fullres.append(get_bin_sample(fulldf.iloc[i*32: (i+1)*32]))
        if i%500==0:
            print(i)
    return pd.concat(fullres, axis=0)
    
# create dataset for given language, score data with appropriate metric
# exsamps is 
def create_sortedbatch_data(lpair, scorer, tok, bsize=32, exsamps=4, cometArch=True):
    lcands = load_cands(exploded_dirs[lpair])
    # set initial sample size to 32, TODO maybe 64 if we need more data
    cand_df = lset_tocominp(lcands, 32, 1)
    bdir = "torchsaved/"+scorer+str(lpair)+"ex"+str(exsamps)+"train"
    # ensure that we're valid, load in old data if needed
    assert scorer=='cqe' or scorer=='comet' or scorer=='bleurt'

    if exists(bdir+".pkl"):
        print("loading from existings")
        with open(bdir+".pkl", "rb") as file:
            return pickle.load(file)
    
    if exists(bdir+".csv"):
        ncand_df = pd.read_csv(bdir+".csv", index_col=0)
    else:
        # rescore data to get labels
        cand_df["score"] = get_scores_auto(cand_df['hyp'], cand_df['src'], cand_df['ref'], scorer, "")
        cand_df = cand_df.sort_values(['ref', "score"]).reset_index().drop(columns=['index'])
        cand_df = cand_df[cand_df["src"].str.contains("&#")==False]
        cand_df = cand_df[cand_df['src'].str.len()>40]
        cand_df = cand_df[cand_df['ref'].str.len()>40]
        ncand_df = convert_bin_sample(cand_df, exsamps)
        ncand_df.to_csv(bdir+".csv")   

    if cometArch:
        print("Using COMET arch, tokenize during collation")
        tinps = list(ncand_df['src']), list(ncand_df['hyp']), list(ncand_df['score'])
        return tinps
    # process data / sort, save again 
    print("TODO SANITY CHECK DATA FOR EACH NEW SETTING")

    print("We have this many candidates: ", len(ncand_df))
    tinps = tokenize_inputs(ncand_df, tok)
    with open(bdir+".pkl", "wb") as file:
        pickle.dump(tinps, file)
    return tinps



