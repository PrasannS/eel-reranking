# file to help with running lattice pipeline on a set of candidates, and generate numbers for tables

import torch
import pickle
import numpy as np
import pandas as pd
import re
from encode_utils.efficient_rerank import run_comstyle

# get token level scores from model, given hypothesis and input source
def get_hyp_sco(inphyp, inpsrc, args):
    tok = args['tok']
    dev = args['device']
    model = args['model']

    # calculate inputs
    tokens = tok(inphyp, return_tensors='pt').to(dev)
    tokens = tokens.input_ids
    positionids = None
    toked_inp = tok([inpsrc], return_tensors="pt").to(dev)
    # get causal mask
    tmpmask = torch.tril(torch.ones(len(tokens[0]), len(tokens[0]))).unsqueeze(0).to(dev)
    # run through model
    predout = model(toked_inp.input_ids, toked_inp.attention_mask, tokens, positionids, \
        tmpmask)
    return predout['score']

# test out reranking multiple EEL outputs (observe improvements)
def lattice_multi_rerank(ind, n, scofunct, afunc, args):
    explode_df = args['explode_df']
    base = args['base']
    goldmetric = args['goldmetric']
    model = args['model']

    # get graph, get the "best candidate"
    graph = pickle.load(open(base+str(ind), 'rb'))
    nexplode = explode_df[explode_df['ref']==graph['ref']].reset_index()
    #print(len(nexplode))
    if len(nexplode)==0:
        return None
    bestcand = np.argmax(list(nexplode[goldmetric]))
    bestcand = nexplode.iloc[bestcand]
    
    goldsco = get_hyp_sco(bestcand['hyp'], bestcand['src'], args)
    goldsco = torch.sum(goldsco[0])
    bpred = -100
    bhyp = ""
    ascos = []
    ahyps = []
    numnodes = 0
    for i in range(n):
        graph = pickle.load(open(base+str(ind), 'rb'))
        # generate with model
        bestpath , flattened, pnodes, mask, sents, posids, pred, _, \
            flnodes, dpath, beplist, besclist, totnodes, bsco = run_comstyle(graph, model, scofunct, "noun", {'afunc':afunc}, True)
        predhyp = bestpath[0][4:]
        hypsco = torch.sum(get_hyp_sco(predhyp, bestcand['src'], args)[0])
        if hypsco>bpred:
            bpred = hypsco
            bhyp = predhyp
        ascos.append(hypsco)
        ahyps.append(predhyp)
        numnodes = len(flattened)
    return bpred, bhyp, goldsco, bestcand['hyp'], ascos, ahyps, numnodes, bestcand['src'], bestcand['ref']

# get multiple things with the lattice, rerank on each (not optimized, so it is a bit slow)
def all_lattice_multi(n, scofunct, afunc, args):
    pdistr = []
    cnt = 0
    SETLEN = args['setlen']
    for i in range(SETLEN):
        try:
            outval = lattice_multi_rerank(i, n, scofunct, afunc, args)
        except:
            print("had an error")
        if outval==None:
            continue
        else:
            print(cnt, " ", i, " ", outval[0], " ", outval[2], " ")
            pdistr.append({
                'hyp':outval[1],
                'hypsco':outval[0],
                'gold':outval[3],
                'goldsco':outval[2],
                'ascos':[float(f) for f in outval[4]],
                'ahyps':outval[5],
                'numnodes':outval[6],
                'src':outval[7],
                'ref':outval[8],
            })
            cnt+=1
    res = pd.DataFrame(pdistr)
    return res

# rerank given a random sample, with respect to target
def all_unnoun_multi(sampsize, metric, args):
    pdistr = []
    cnt = 0
    SETLEN = args['setlen']
    base = args['base']
    explode_df = args['explode_df']
    for i in range(SETLEN):
        try:
            graph = pickle.load(open(base+str(i), 'rb'))
        except:
            break
        nexplode = explode_df[explode_df['ref']==graph['ref']].reset_index()
        if len(nexplode)==0:
            continue
        if sampsize>0:
            nexplode = nexplode.sample(n=sampsize)
            assert len(nexplode)==sampsize
        bestcand = np.argmax(list(nexplode[metric]))
        bestcand = nexplode.iloc[bestcand]

        goldsco = get_hyp_sco(bestcand['hyp'], bestcand['src'], args)
        goldsco = torch.sum(goldsco[0])
        pdistr.append(goldsco)
    
    return pdistr

def mean(l):
    l = list(l)
    if type(l[0]) is str:
        l = [float(re.findall("\d+\.\d+", lent)[0]) for lent in l]
    return sum(l)/len(l)
