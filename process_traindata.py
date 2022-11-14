import json
import random
from mt_scores import get_scores_auto
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

# create dataset for given language, score data with appropriate metric
def create_sortedbatch_data(lpair, scorer, tok, bsize=32, exsamps=2, isTrain=True):
    lcands = load_cands(exploded_dirs[lpair])
    cand_df = lset_tocominp(lcands, bsize, exsamps)
    bdir = "torchsaved/"+scorer+str(lpair)+"ex"+str(exsamps)+"train"
    # ensure that we're valid, load in old data if needed
    assert scorer=='cqe' or scorer=='comet' or scorer=='bleurt'

    if exists(bdir+".pkl"):
        print("loading from existings")
        with open(bdir+".pkl", "rb") as file:
            return pickle.load(file)
    
    if exists(bdir+".csv"):
        cand_df = pd.read_csv(bdir+".csv", index_col=0)
    else:
        # rescore data to get labels
        cand_df["score"] = get_scores_auto(cand_df['hyp'], cand_df['src'], cand_df['ref'], scorer, "")
        cand_df.to_csv(bdir+".csv")
    
    # process data / sort, save again 
    cand_df = cand_df.sort_values(['ref', "score"]).reset_index().drop(columns=['index'])
    cand_df = cand_df[cand_df["src"].str.contains("&#")==False]
    cand_df = cand_df[cand_df['src'].str.len()>40]
    cand_df = cand_df[cand_df['ref'].str.len()>40]
    cand_df.to_csv(bdir+".csv")
    print("We have this many candidates: ", len(cand_df))
    tinps = tokenize_inputs(cand_df, tok)
    with open(bdir+".pkl", "wb") as file:
        pickle.dump(tinps, file)
    return tinps



