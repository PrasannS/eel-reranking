# file to help with running lattice pipeline on a set of candidates, and generate numbers for tables

import torch
import pickle
import numpy as np
import pandas as pd
import re
import time
from encode_utils.efficient_rerank import run_comstyle_multi

# get token level scores from model, given hypothesis and input source
def get_hyp_sco(inphyp, inpsrc, args):
    tok = args['tok']
    dev = args['device']
    model = args['model']

    # calculate inputs
    tokens = tok(inphyp, return_tensors='pt', truncation=True).to(dev)
    tokens = tokens.input_ids
    positionids = None
    toked_inp = tok([inpsrc], return_tensors="pt").to(dev)
    # get causal mask
    tmpmask = torch.tril(torch.ones(len(tokens[0]), len(tokens[0]))).unsqueeze(0).to(dev)
    # run through model
    predout = model(toked_inp.input_ids, toked_inp.attention_mask, tokens, positionids, \
        tmpmask)
    return predout['score']

# TODO put elsewhere - make batched version of get_hyp_sco (TODO also do dataloader setup or smth)
def causalmask (a, dev):
    masksdef = torch.zeros((a.shape[0], a.shape[1],a.shape[1]), device=dev)
    for i in range(len(a)):
        lim = int(torch.sum(a[i]))
        masksdef[i, :lim, :lim] = torch.tril(torch.ones((lim, lim)))
    return masksdef

# get scores given a batch of src, hypothesis pairs
def batch_hyp_sco(srcs, hyps, args):
    tok = args['tok']
    dev = args['device']
    model = args['model']
    
    out_toks = tok(hyps, return_tensors='pt', padding=True, truncation=True).to(dev)
    out_tokens = out_toks.input_ids
    hypmask = causalmask(out_toks.attention_mask, dev)
    
    positionids = None
    toked_inp = tok(srcs, return_tensors="pt", padding=True, truncation=True).to(dev)
    
    predout = model(toked_inp.input_ids, toked_inp.attention_mask, out_tokens, positionids, \
        hypmask)
    
    return torch.sum(predout['score'], 1)#, toked_inp, out_tokens, positionids, hypmask


# get multiple things with the lattice, rerank on each (not optimized, so it is a bit slow)
def all_timing_experiment(scofunct, afunc, args):
    pdistr = []
    cnt = 0
    SETLEN = args['setlen']
    for i in range(SETLEN):
        print(i)
        outval = lattice_timing_experiment(i, scofunct, afunc, args)
        #except:
        #print("had an error")
        if outval==None:
            continue
        else:
            pdistr.append(outval)
            cnt+=1
    res = pd.DataFrame(pdistr)
    return res

def lattice_timing_experiment(ind, scofunct, afunc, args):
    explode_df = args['explode_df']
    base = args['base']
    goldmetric = args['goldmetric']
    model = args['model']
    nounmode = "noun" in goldmetric

    graph = pickle.load(open(base+str(ind), 'rb'))
    nexplode = explode_df[explode_df['ref']==graph['ref']].reset_index()

    if len(nexplode)==0:
        return None
    
    ehyps = nexplode['hyp'][:48]
    if len(ehyps)!=48: # can't do timing comparison without enough cands
        return None
    srcs = [nexplode['src'][0]]*8
    ind = 0
    stime = time.time()
    time8, time32, time48 = 0, 0, 0
    # Run experiment on 6 batches of 8 
    while ind < 6:
        htmp = ehyps[ind*8:(ind+1)*8]
        batch_hyp_sco(list(srcs), list(htmp), args)
        if ind==0:
            time8 = time.time() - stime
        elif ind==3:
            time32 = time.time() - stime
        elif ind==5:
            time48 = time.time() - stime
        ind+=1

    # lattice time 1
    bestpath , flattened, pnodes, mask, sents, posids, pred, _, \
        flnodes, dpath, beplist, besclist, totnodes, bsco, timelat1 = run_comstyle_multi(graph, model, scofunct, goldmetric, {'afunc':afunc}, True, 1)
    
    # lattice time 8
    bestpath , flattened, pnodes, mask, sents, posids, pred, _, \
            flnodes, dpath, beplist, besclist, totnodes, bsco, timelat8 = run_comstyle_multi(graph, model, scofunct, goldmetric, {'afunc':afunc}, True, 8)
    
    # 1 batch of 12
    stime = time.time()
    batch_hyp_sco([nexplode['src'][0]]*12, list(nexplode['hyp'][:12]), args)
    time12 = time.time() - stime

    return {
        'time8':time8,
        'time12':time12,
        'time32':time32,
        'time50':time48,
        'timelat1':timelat1,
        'timelat8':timelat8
    }
    


# test out reranking multiple EEL outputs (observe improvements)
def lattice_multi_rerank(ind, n, scofunct, afunc, args):
    explode_df = args['explode_df']
    base = args['base']
    goldmetric = args['goldmetric']
    model = args['model']
    nounmode = "noun" in goldmetric

    # get graph, get the "best candidate"
    graph = pickle.load(open(base+str(ind), 'rb'))
    nexplode = explode_df[explode_df['ref']==graph['ref']].reset_index()

    if len(nexplode)==0:
        return None
    bestcand = np.argmax(list(nexplode[goldmetric]))
    bestcand = nexplode.iloc[bestcand]
    # TODO should be able to just use scores now? maybe assert at some point
    if args['efficient_eval']:
        goldsco = 0
    else:
        if nounmode:
            goldsco = get_hyp_sco(bestcand['hyp'], "noun", args)
        else:
            goldsco = get_hyp_sco(bestcand['hyp'], bestcand['src'], args)
        goldsco = torch.sum(goldsco[0])

    numnodes = 0
    graph = pickle.load(open(base+str(ind), 'rb'))
    # generate with model
    bestpath , flattened, pnodes, mask, sents, posids, pred, _, \
        flnodes, dpath, beplist, besclist, totnodes, bsco, timedict = run_comstyle_multi(graph, model, scofunct, goldmetric, {'afunc':afunc, 'defpos':args['defpos']}, True, n)
    assert graph['input']==bestcand['src']
    # get all appropriate inps
    ahyps = [bp[4:] for bp in bestpath]
    if nounmode:
        srcs = ["noun"]*len(ahyps)
    else:
        srcs = [graph['input']]*len(ahyps)
    # TODO there may be issue from posid degradation?
    if args['efficient_eval']:
        ascos = bsco
        # Rescore everything, send back result (TODO can we use dp scores instead?)
    else:
        #print("e")
        ascos = batch_hyp_sco(srcs, ahyps, args)
    ascos = [float(a) for a in ascos]
    numnodes = len(flattened)
    bestind = np.argmax(list(ascos))

    return ascos[bestind], ahyps[bestind], goldsco, bestcand['hyp'], ascos, ahyps, numnodes, bestcand['src'], bestcand['ref'], bsco, timedict

# get multiple things with the lattice, rerank on each (not optimized, so it is a bit slow)
def all_lattice_multi(n, scofunct, afunc, args):
    pdistr = []
    cnt = 0
    SETLEN = args['setlen']
    for i in range(SETLEN):
        #try:
        outval = lattice_multi_rerank(i, n, scofunct, afunc, args)
        #except:
        #print("had an error")
        if outval==None:
            continue
        else:
            print(cnt, " ", i, " ", outval[0], " ", outval[2], " ")
            pdistr.append({
                'hyp':outval[1],
                'hypsco':float(outval[0]),
                'gold':outval[3],
                'goldsco':float(outval[2]),
                'ascos':[float(f) for f in outval[4]],
                'ahyps':outval[5],
                'numnodes':outval[6],
                'src':outval[7],
                'ref':outval[8],
                'origscos':[float(f) for f in outval[9]],
                'timing': outval[10]
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
    nounmode = "noun" in args['goldmetric']
    for i in range(SETLEN):
        try:
            graph = pickle.load(open(base+str(i), 'rb'))
        except:
            break
        nexplode = explode_df[explode_df['ref']==graph['ref']].reset_index()
        if len(nexplode)==0:
            continue
        running = True
        # keep on sampling if there are huge cands? 
        while running:
            bestcand = None
            try:
                if sampsize>0:
                    nexplode = nexplode.sample(n=sampsize)
                    assert len(nexplode)==sampsize
                bestcand = np.argmax(list(nexplode[metric]))
                bestcand = nexplode.iloc[bestcand]
                if nounmode:
                    goldsco = get_hyp_sco(bestcand['hyp'], "noun", args)
                else:
                    goldsco = get_hyp_sco(bestcand['hyp'], bestcand['src'], args)
                goldsco = torch.sum(goldsco[0])
                pdistr.append(goldsco)
                # dummy test to get timing numbers, TODO remove
                if args['noregen']:
                    print("h")
                    for i in range(sampsize-1):
                        goldsco = get_hyp_sco(bestcand['hyp'], "noun", args)
                running = False
            except:
                print(bestcand['hyp'])
                continue
        
        print(i)
    
    return pdistr

def mean(l):
    l = list(l)
    if type(l[0]) is str:
        l = [float(re.findall("\d+\.\d+", lent)[0]) for lent in l]
    return sum(l)/len(l)
