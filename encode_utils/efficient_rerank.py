import torch
from .new_mask_utils import get_causal_mask
from .encoding_utils import create_inputs

import torch.nn as nn
from transformers import AutoModel
from .new_flatten_lattice import get_dictlist, detok
from .sco_funct import onlyprob
import time

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

xlm_tok = detok

import torch.nn as nn
from transformers import AutoModel

import sys
sys.path.insert(1, '/mnt/data1/prasann/latticegen/lattice-generation/COMET')

from tfr_models.comet.models import load_from_checkpoint as lfc

# Returns Token Scores For Each (Sum becomes Regression)
class XLMCometEmbeds(nn.Module):
    
    def __init__(self, drop_rate=0.1):
        # TODO should we be freezing layers?
        super().__init__()
        
        self.xlmroberta = AutoModel.from_pretrained('xlm-roberta-base')
        # Num labels 1 should just indicate regression (?)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.xlmroberta.config.hidden_size, 1), 
        )
        self.to(device)
        
    def forward(self, input_ids, positions, attention_masks):
        # don't finetune xlmroberta model
        #with torch.no_grad():
        if positions==None:
            word_rep, sentence_rep = self.xlmroberta(input_ids, attention_mask=attention_masks, encoder_attention_mask=attention_masks, return_dict=False)
        else:
            word_rep, sentence_rep = self.xlmroberta(input_ids, position_ids = positions, attention_mask=attention_masks, encoder_attention_mask=attention_masks, return_dict=False)

        # use the first <s> token as a CLS token, TODO experiment with using the sum of 
        # ensure padding not factored in
        #word_rep = word_rep*(input_ids>0).unsqueeze(-1)
        
        word_rep = word_rep*(input_ids>0).unsqueeze(-1)
        #outputs = self.regressor(torch.sum(word_rep, 1))
        #print("Shape: ", outputs.shape)
        lastval = torch.inner(word_rep, self.regressor[1].weight).squeeze(-1)
        return lastval, self.regressor(torch.sum(word_rep, 1))

def check_accuracy(setpred, setlabels):
    cor = 0
    tot = 0
    for i in range(0, len(setpred)):
        ex = setpred[i]
        for j in range(0, len(ex)):
            if sum(setlabels[i][j])==0:
                continue
            elif torch.argmax(setlabels[i][j])==0:
                continue
            tot+=1
            if torch.argmax(ex[j])==torch.argmax(setlabels[i][j]):
                cor+=1
    return cor/tot

# correct posids
def mod_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            if p[i]==0:
                p[i] = i
    return cop

# set posids to default
def def_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            p[i] = i
    return cop

def prepend_input(pgraph, inp):
    
    inptoks = xlm_tok(inp).input_ids
    # add in the <s> token
    # we don't need the </s> token by how the mask works
    # inptoks.append(0)
    posadd = len(inptoks)
    inpflat = []
    ind = 0
    for i in range(len(inptoks)):
        nl = []
        inp = inptoks[i]
        if i<(len(inptoks)-1):
            nl.append(str(inptoks[i+1])+" "+str(ind+1))
        inpflat.append({
            'token_idx':inp,
            'pos':ind,
            'id': str(inp)+" "+str(ind),
            'nexts': nl,
            'score': 1,
        })
        ind+=1
    inpflat[-1]['nexts'].append(pgraph[0]['id'])
    
    inpflat.extend(pgraph)
    inpflat[posadd]['token_idx']=0
    inpflat[posadd]['token_str']= "<s>"
    for i in range(posadd, len(inpflat)):
        extok = inpflat[i]
        extok['pos']+=posadd
        
    return inpflat, posadd

# topological sort the graphs, make sure that nodes that are next always come next in the list
def topo_sort_nodes(pgraph):
    cp = [p for p in pgraph]
    tmpgraph = {}
    for c in cp:
        tmpgraph[c.uid]=c
    res = []
    visited = []
    # reverse ordering
    topo_recurse(cp[0].uid, res, [], tmpgraph)
    # don't reverse anymore
    # res.reverse()
    return res
        
def topo_recurse(curid, toplist, visited, graph):
    if curid in visited:
        return 
    # for stuff in truncated part of graph TODO
    if curid not in graph.keys():
        return
    node = graph[curid]
    visited.append(curid)
    for nid in [n.uid for n in node.nextlist]:
        topo_recurse(nid, toplist, visited, graph)
    # once done, add to the beginnings
    toplist.insert(0, node)

# take in list of truncnodes, sort for dynamic programming
def prepare_nodes(truncnodes, scores, padd):
    ind = 0

    #for p in range(len(truncnodes)):
    # get rid of this later if we decide to have multiple canvi
    #    assert len(truncnodes[p])<=512
    for i in range(min(len(truncnodes[0]),512)):
        # get score after offset (we only get for decoded stuff)
        # TODO do some kind of assert
        truncnodes[0][i].score = scores[i+padd]
    
    result = [topo_sort_nodes(truncnodes[0])]
    #for trunc in truncnodes:
    #   result.append())    
    for trunc in result:
        dpos = 0
        for node in trunc:
            node.dppos = dpos
            dpos+=1
            
    return result

MINPROP = 0.5
MINDEF = -10000
def dynamic_path(prepnodes, sco_funct, posapp, usednodes, norm):
    
    bplist = [None]*len(prepnodes)
    bscolist = [MINDEF]*len(prepnodes)
    endings = []
    # go through topologically sorted list of nodes, update best path for each
    for prep in prepnodes:
        if bplist[prep.dppos]==None:
            bplist[prep.dppos] = []
        if len(prep.prevs)>0:
            mval = MINDEF
            mprev = None
            for p in prep.prevs:
                if p.dppos>=0 and bscolist[p.dppos]>mval:
                    mval = bscolist[p.dppos]
                    mprev = p
            if mprev is not None:
                # use from previous
                bplist[prep.dppos].extend(bplist[mprev.dppos])
                bscolist[prep.dppos] = bscolist[mprev.dppos] + sco_funct(prep, usednodes, norm)

        # TODO look into endings that are happening due to excessive trunction
        ncnt = 0
        # TODO more thorough check
        for prn in prep.nextlist:
            if prn in prepnodes:
                ncnt+=1
        if ncnt==0:
            endings.append(prep.dppos)
        if bscolist[prep.dppos]==MINDEF:
            bscolist[prep.dppos]= sco_funct(prep, usednodes, norm)
        bplist[prep.dppos].append(prep)

    bestpath = []
    bestsco = MINDEF
    for e in endings:
        # make sure we hit minimum length proportional to input
        if len(bplist[e])>(MINPROP)*posapp:
            if (bscolist[e]/len(bplist[e]))>bestsco:
                bestsco = bscolist[e]/len(bplist[e])
                bestpath = bplist[e]
    if len(bestpath)==0:
        lenlist = [len(bplist[e]) for e in endings]
        bind = lenlist.index(max(lenlist))
        bestsco = bscolist[endings[bind]]
        bestpath = bplist[endings[bind]]
        print("suboptimal, ", [tnode.token_str for tnode in bestpath])
    return bestpath, bestsco, bplist, bscolist

def get_effrerank_model(keystr, evalbool=True):
    reflessmod = None
    if 'parent' in keystr:
        print("USING TABLE->TEXT MODEL")
        reflessmod = lfc("/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_57/checkpoints/epoch=1-step=50000.ckpt", evalbool).to(device)
    if keystr == "comstyle":
        print("USING MT MODEL")
        # TODO update to most updated model at a given time
        reflessmod = lfc("/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_43/checkpoints/epoch=3-step=130000.ckpt", evalbool).to(device)
    elif keystr == "comnocause":
        reflessmod = lfc("/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_38/checkpoints/epoch=3-step=140000.ckpt", evalbool).to(device)
    elif keystr == "noun":
        print("NOUN MODEL")
        reflessmod = lfc("/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_44/checkpoints/epoch=9-step=40000.ckpt", evalbool).to(device)
        
    reflessmod.eval()
    return reflessmod

MAX_TOKS = 512
# run pipeline, but we use comet-style model 
def run_comstyle_multi(graph, model, scofunct, outfile, params, extra=False, nummasks=1, verbose=False):
    # get converted, flattened lattice

    nodetime = time.time()
    flattened, flnodes = get_dictlist(graph, True)
    nodetime =  time.time() - nodetime
    totnodes = len(flnodes)
    enctime = 0
    masktime = 0
    blist = []
    slist = []
    
    msks = []
    # don't do compute if we're just getting adhoc baseline
    if scofunct is not onlyprob:

        # make sure that we're only working with the tokens that fit into canvas
        truncflat = flattened[:MAX_TOKS]
        sents, posids = create_inputs([truncflat])
        # NOTE this is not the same for BERT
        if "pqe" not in outfile:
            posids = posids + 2 # don't add posids for table2text since it already does that
        
        # type of model is ReflessEval, input format of (everything)
        if "noun" in outfile:
            toked_inp = xlm_tok(["noun"], return_tensors="pt").to(device)
        else:
            toked_inp = xlm_tok([graph['input']], return_tensors="pt").to(device)

        
        masktime = time.time()
        # get n shots worth of masks
        for i in range(nummasks):
            msks.append(get_causal_mask(flnodes, 0, params, False))
        masktime = time.time() - masktime
        # src_input_ids, src_attention_mask, mt_input_ids, mt_pos_ids, mt_attention_mask
        assert len(sents.shape)==2
        assert len(toked_inp.input_ids.shape)==2
        inpshape = nummasks, toked_inp.input_ids.shape[1]
        hypshape = nummasks, sents.shape[1]
        with torch.no_grad():
            # TODO can make more efficient by shifting .to calls
            # TODO increase efficiency with batching (also ensure mask gen on device)
            # pass everything in one batch
            enctime = time.time()
            posids = posids.expand(hypshape)
            if params['defpos']:
                posids = None
            predout = model(toked_inp.input_ids.expand(inpshape), toked_inp.attention_mask.expand(inpshape) \
                        , sents.expand(hypshape), posids, torch.stack(msks).to(device))
            enctime = time.time() - enctime
            # undo normalization since that happens later anyways
            pred = (predout['score'] * predout['norm']).cpu()
            norm = predout['norm']
    else:
        pred = [0]*512
        pred = [pred]
        norm = len(flnodes)
        sents = None
        posids = None
        totnodes = norm
        
    usednodes = []
    
    #dptime = time.time()
    preptime, dptime, otherdp = 0, 0, 0
    # go through all sets of scores, do dynamic programming on each (TODO can we, should we simplify further)
    for p in range(len(pred)):
        preptime -= time.time()
        # multiple rounds for diverse decoding, TODO time optimization (don't fill up canvas) if not already?
        pnodes = prepare_nodes([flnodes[:512]], pred[p], 0)
        preptime += time.time()
        dptime -= time.time()
        dpath, bsco, beplist, besclist = dynamic_path(pnodes[0], scofunct, 0, usednodes, norm)
        dptime += time.time()
        otherdp -= time.time()
        usednodes.extend([dp for dp in dpath])
        blist.append(xlm_tok.decode([dp.token_idx for dp in dpath]))
        slist.append(bsco)
        otherdp+= time.time()
        
    times = {
        'dp':dptime,
        'dp_prep': preptime,
        "dp_other": otherdp,
        'mask': masktime,
        'encode': enctime,
        'preprocess': nodetime
    }
    # verbose return for debugging
    if extra:
        return blist, flattened, pnodes, msks, sents, posids, pred, 0, flnodes, dpath, beplist, besclist, totnodes, slist, times
    return blist

MAX_TOKS = 512
# run pipeline, but we use comet-style model 
def run_comstyle(graph, model, scofunct, outfile, params, extra=False, numruns=1, verbose=False):

    flattened, flnodes = get_dictlist(graph, True)

    totnodes = len(flnodes)

    # since we don't have any pre-input, we can just start from beginning?
    # TODO validate that there isn't weirdness here
    mask = get_causal_mask(flnodes, 0, params, False)
    # make sure that we're only working with the tokens that fit into canvas
    truncflat = flattened[:MAX_TOKS]
    sents, posids = create_inputs([truncflat])
    # TODO change this back for bert?
    posids = posids + 2
    # type of model is ReflessEval, input format of (everything)
    if "noun" in outfile:
        toked_inp = xlm_tok(["noun"], return_tensors="pt").to(device)
    else:
        toked_inp = xlm_tok([graph['input']], return_tensors="pt").to(device)
    # src_input_ids, src_attention_mask, mt_input_ids, mt_pos_ids, mt_attention_mask
    with torch.no_grad():
        # TODO can make more efficient by shifting .to calls
        predout = model(toked_inp.input_ids, toked_inp.attention_mask, sents, posids, \
            mask.unsqueeze(0).to(device))
        pred = predout['score']
        norm = predout['norm']

    blist = []
    slist = []
    usednodes = []
    # multiple rounds for diverse decoding
    for decround in range(numruns):
        # TODO make sure format's still fine
        pnodes = prepare_nodes([flnodes[:512]], pred[0], 0)
        dpath, bsco, beplist, besclist = dynamic_path(pnodes[0], scofunct, 0, usednodes, norm)

        usednodes.extend([dp for dp in dpath])
        blist.append(xlm_tok.decode([dp.token_idx for dp in dpath]))
        slist.append(bsco)
    if verbose:
        print("SRC - "+graph['input'])
        print("PRED - "+blist[0])
        print("REF - "+graph['ref'])
    # verbose return for debugging
    if extra:
        return blist , flattened, pnodes, mask, sents, posids, pred, 0, flnodes, dpath, beplist, besclist, totnodes, slist
    return blist

def run_pipeline(graph, model, scofunct, extra=False, numruns=1, verbose=False):
    #flatold = fl.flatten_lattice(graph)
    flattened, flnodes = get_dictlist(graph, True)
    totnodes = len(flnodes)
    ppinput = prepend_input(flattened, graph['input'])
    flattened = ppinput[0]
    # covered = fl.get_cover_paths(flattened)

    posadd = ppinput[1]
    mask = get_causal_mask(flnodes, posadd)
    # make sure that we're only working with the tokens that fit into canvas
    truncflat = flattened[:512]
    sents, posids = create_inputs([truncflat])
    # TODO change this for bert
    posids = posids + 2
    with torch.no_grad():
        pred = model(sents, posids, mask.unsqueeze(0).to(device))
    # fls = [truncflat]
    #prepared_pgraphs = prepare_pgraphs(fls, pred[0])
    #bestpath = dp_pgraph(prepared_pgraphs[0], scofunct)
    #best = xlm_tok.decode(bestpath)
    blist = []
    slist = []
    usednodes = []
    for decround in range(numruns):
        pnodes = prepare_nodes([flnodes[:512-(posadd)]], pred[0], posadd)
        dpath, bsco, beplist, besclist = dynamic_path(pnodes[0], scofunct, posadd, usednodes)
        usednodes.extend([dp for dp in dpath])
        blist.append(xlm_tok.decode([dp.token_idx for dp in dpath]))
        slist.append(bsco)
    if verbose:
        print("SRC - "+graph['input'])
        print("PRED - "+blist[0])
        print("REF - "+graph['ref'])
    # verbose return for debugging
    if extra:
        return blist, flattened, pnodes, mask, sents, posids, pred, posadd, flnodes, dpath, beplist, besclist, totnodes, slist
    return blist



