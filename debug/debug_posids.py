import pickle
from encode_utils.rerank_data import rerank_dist, rerank_single
from encode_utils.efficient_rerank import get_effrerank_model, run_comstyle
from encode_utils.sco_funct import weightaddprob, default_scofunct
from encode_utils.mt_scores import get_scores_auto
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch
import random
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# score check (at token level) the best option, and the new option (should be same until they get to non-identical tokens)

# get token level scores from model
def get_hyp_sco_verb(inphyp, posids=None):
    
    tokens = xlm_tok(inphyp, return_tensors='pt').to(device)
    tokens = tokens.input_ids
    #print(inphyp)
    #print(tokens)
    
    if posids is None: 
        mask = tokens.ne(-1).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + 0) * mask
        positionids = incremental_indices.long() + 1
    elif -1 in posids:
        positionids = None
    else:
        # get token at the end
        positionids = torch.tensor(posids+[posids[-1]+1]).unsqueeze(0).to(device)
    #print(positionids)
    tmpmask = torch.tril(torch.ones(len(tokens[0]), len(tokens[0]))).unsqueeze(0).to(device)
    #print(tokens.shape)
    #print(positionids.shape)
    #print()
    #print(tokens.shape)
    #print(torch.max(positionids))
    toked_inp = xlm_tok(["noun"], return_tensors="pt").to(device)
    predout = encodemod(toked_inp.input_ids, toked_inp.attention_mask, tokens, positionids, \
        tmpmask)
    tmppred = predout['score']
    #norm = predout['norm']
    return tmppred, tokens, positionids, tmpmask

def randomsingle(mask, row, checknodes, mlen):
    if row>0:
        avail = []
        # use next with highest prob
        for n in checknodes:
            if n.canvpos<mlen: # keep within bounds
                avail.append(n)
        if len(avail)==0:
            print(len(checknodes))
            print(row)
        mask[row][random.choice(avail).canvpos] = 1

# sanity check that masks are functioning in an ok way (passed)
def mask_sanity(msk, inps, posids):
    allinps = []
    allpos = []
    for m in msk:
        # get spare
        tmptoks = []
        tmppos = []
        for t in range(len(m)):
            if m[t]!=0:
                tmptoks.append(inps[t])
                tmppos.append(posids[t])
        resort = sorted(zip(tmppos, tmptoks))
        tmptoks = [x for _,x in resort]
        tmppos = [x for x, _ in resort]
        allinps.append(torch.tensor(tmptoks).int())
        allpos.append(tmppos)
    return allinps, allpos


if __name__=="__main__":
    
    # load the model
    encodemod = get_effrerank_model("noun")
    xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # get graph to examine
    ind = 3
    base = "outputs/graph_pickles/frenchbeam12_reversed/"
    #base = "outputs/graph_pickles/frtest_reversed/"
    graph = pickle.load(open(base+str(ind), 'rb'))
    bestpath , flattened, pnodes, mask, sents, posids, pred, _, \
        flnodes, dpath, beplist, besclist, totnodes, bsco = run_comstyle(graph, encodemod, default_scofunct, "noun", {'afunc':randomsingle}, True)
    predhyp = bestpath[0][4:]

    ainps, apos = mask_sanity(mask, sents[0], posids[0])
    testsens = xlm_tok.batch_decode(ainps)

    index = 40
    #print(apos[index])
    p_pred, p_tok, p_pos, p_mask = get_hyp_sco_verb(testsens[index][4:], [-1])
    n_pred, n_tok, n_pos, n_mask = get_hyp_sco_verb(testsens[index][4:], None)