import pickle
from encode_utils.rerank_data import rerank_dist, rerank_single
from encode_utils.efficient_rerank import get_effrerank_model, run_comstyle
from encode_utils.sco_funct import weightaddprob, default_scofunct
from encode_utils.mt_scores import get_scores_auto
from encode_utils.new_flatten_lattice import get_dictlist
from encode_utils.new_mask_utils import randomsingle, useall
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch
from generate_tables import metrics_mapping
import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# setup / load model
xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

mb_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

# recursive method to sanity check first (assume no recomb, no cycles)
def explode_path(ind, processed=False):
    graph = pickle.load(open(base+str(ind), 'rb'))
    allpaths = []
    if processed:
        fldict, flnodes = get_dictlist(graph, True)
        explode_helper([], flnodes[0], allpaths, xlm_tok)
    else:
        explode_helper([], graph['root'], allpaths, mb_tok)
        allpaths = [a[10:] for a in allpaths]
    nexplode = noun_explode[noun_explode['ref']==graph['ref']].reset_index()
    return allpaths, list(nexplode['hyp'])

# helper for exploding paths
def explode_helper(prevpath, node, apaths, tok):
    prevpath.append(node.token_idx)
    if len(node.nextlist)==0:
        apaths.append(mb_tok.decode(prevpath))
    else:
        for n in node.nextlist:
            explode_helper(prevpath, n, apaths, tok)
    prevpath.pop()

"""
# get token level scores from model
def get_hyp_sco(inphyp, posids=None):
    
    tokens = xlm_tok(inphyp, return_tensors='pt').to(device)
    tokens = tokens.input_ids
    if posids is None: 
        positionids = None
    else:
        # get token at the end
        positionids = torch.tensor(posids+[posids[-1]+1]).unsqueeze(0).to(device)
    tmpmask = torch.tril(torch.ones(len(tokens[0]), len(tokens[0]))).unsqueeze(0).to(device)

    toked_inp = xlm_tok(["noun"], return_tensors="pt").to(device)
    predout = encodemod(toked_inp.input_ids, toked_inp.attention_mask, tokens, positionids, \
        tmpmask)
    tmppred = predout['score']
    #norm = predout['norm']
    return tmppred
"""

# get token level scores from model, given hypothesis and input source
def get_hyp_sco(inphyp, inpsrc, args):
    tok = args['tok']
    dev = args['device']
    model = args['model']

    # calculate inputs
    tokens = tok(inphyp, return_tensors='pt', truncation=True).to(dev)
    tokens = tokens.input_ids
    positionids = None
    toked_inp = tok(inpsrc, return_tensors="pt").to(dev)
    # get causal mask
    tmpmask = torch.tril(torch.ones(len(tokens[0]), len(tokens[0]))).unsqueeze(0).to(dev)
    # run through model
    predout = model(toked_inp.input_ids, toked_inp.attention_mask, tokens, positionids, \
        tmpmask)
    return predout['score']


# TODO do a validation that old score generation way and current have same bests
def get_ind_result(ind):
    graph = pickle.load(open(base+str(ind), 'rb'))
    texplode = noun_explode[noun_explode['ref']==graph['ref']].reset_index()
    # recalculate noun scores for all
    cscos = []
    for t in list(texplode['hyp']):
        hs = get_hyp_sco(t)
        cscos.append(torch.sum(get_hyp_sco(t)))#*(hs.shape[1]-1))
    print("scodist - ", [float(f) for f in cscos])
    print("max - ", float(max(cscos)))
    bestpath , flattened, pnodes, mask, sents, posids, pred, _, \
            flnodes, dpath, beplist, besclist, totnodes, bsco = run_comstyle(graph, encodemod, default_scofunct, "noun", {'afunc':randomsingle}, True)
    predhyp = bestpath[0][4:]
    ph = get_hyp_sco(predhyp)
    predsco = torch.sum(ph)#*(ph.shape[1]-1)
    print("pred - ", float(predsco))
    print("predhyp")
    return mask, sents, posids, pred, list(texplode['hyp'])

if __name__=="__main__":
    # TODO do this for everything
    mtb12 = pd.read_csv("outputs/score_csvs/mtfrenbeam12v2.csv", index_col=0)
    encodemod = get_effrerank_model("comstyle")
    encodemod.eval()
    xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    argsinp = {
        'tok':xlm_tok,
        'model':encodemod,
        'device':device
    }
    #encodemod.predict()
    tmpset = mtb12.loc[:3]
    tmpset = tmpset.rename(columns={'dupcqe':"dco2"})

    metrics_mapping("dupcqe", tmpset)
    inex = tmpset.loc[0]
    sco = get_hyp_sco(inex['hyp'], inex['src'], argsinp)
    print(torch.sum(sco))

    """
    NOUN MODEL CHECK
    base = "outputs/graph_pickles/frtest_reversed/"
    noun_explode = pd.read_csv("outputs/score_csvs/nounlargeexplodev1.csv")

    base = "outputs/graph_pickles/frtest_reversed/"
    noun_explode = pd.read_csv("outputs/score_csvs/nounlargeexplodev1.csv")

    # setup / load model
    encodemod = get_effrerank_model("noun")
    xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    

    smallset = noun_explode.loc[:3]
    smallset['utnounold'] = list(smallset['utnoun'])
    smallset = smallset.drop(columns=['utnoun'])

    metrics_mapping('utnoun', smallset)

    newsco = torch.sum(get_hyp_sco(smallset['hyp'][0]))

    print("here now")
    """
    

    

    # postproc_paths, orig_paths = explode_path(17, True)    
    # msk, snts, pids, scos, hyps = get_ind_result(17)
    # print(len(pids[0]))


