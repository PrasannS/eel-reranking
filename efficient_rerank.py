from asyncio import new_event_loop
import flatten_lattice as fl
import torch
from encoding_utils import *
from new_mask_utils import get_causal_mask

import torch.nn as nn
from transformers import AutoModel
from new_flatten_lattice import get_dictlist

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

from mask_utils import *
from encoding_utils import *

mbart_tok = fl.bert_tok
xlm_tok = mbart_tok

import torch.nn as nn
from transformers import AutoModel
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
        #extok['id']= str(extok['token_idx'])+" "+str(extok['pos'])
        #for j in range(len(extok['nexts'])):
        #    newpos = extok["pos"]
        #    extok['nexts'][j] = extok['nexts'][j].split()[0]+" "+str(newpos)
    return inpflat, posadd

def causal_mask (pgraph, padd):
    start = connect_mat(pgraph)
    start[:, :padd] = 1
    start[:padd, padd:] = 0 
    start[padd:, padd:] = torch.tril(start[padd:, padd:])
    return start

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
    for p in range(len(truncnodes)):
        # get rid of this later if we decide to have multiple canvi
        assert len(truncnodes[p])<=512
        for i in range(min(len(truncnodes[p]),512)):
            # get score after offset (we only get for decoded stuff)
            # TODO do some kind of assert
            truncnodes[p][i].score = scores[p][i+padd]
    
    result = []
    for trunc in truncnodes:
        result.append(topo_sort_nodes(trunc))
        
    for trunc in result:
        dpos = 0
        for node in trunc:
            node.dppos = dpos
            dpos+=1
            
    return result

MINPROP = 0.5
MINDEF = -10000
def dynamic_path(prepnodes, sco_funct, posapp, usednodes):
    
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
                bscolist[prep.dppos] = bscolist[mprev.dppos] + sco_funct(prep, usednodes)
                #print(bscolist[prep.dppos])
        # TODO look into endings that are happening due to excessive trunction
        ncnt = 0
        # TODO more thorough check
        for prn in prep.nextlist:
            if prn in prepnodes:
                ncnt+=1
        if ncnt==0:
            endings.append(prep.dppos)
        if bscolist[prep.dppos]==MINDEF:
            bscolist[prep.dppos]= sco_funct(prep, usednodes)
        bplist[prep.dppos].append(prep)
        #bscolist[prep.dppos] += prep.score
    
    print("maxend ,", max([len(bplist[e]) for e in endings ]))
    #print([float(bscolist[e]) for e in endings])
    bestpath = []
    bestsco = MINDEF
    for e in endings:
        # make sure we hit minimum length proportional to input
        if len(bplist[e])>(MINPROP)*posapp:
            if bscolist[e]>bestsco:
                bestsco = bscolist[e]
                bestpath = bplist[e]
    if len(bestpath)==0:
        lenlist = [len(bplist[e]) for e in endings]
        bind = lenlist.index(max(lenlist))
        bestsco = bscolist[endings[bind]]
        bestpath = bplist[endings[bind]]
        print("suboptimal, ", [tnode.token_str for tnode in bestpath])
    return bestpath, bplist, bscolist

MAX_TOKS = 512
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
    with torch.no_grad():
        pred = model(sents, posids, mask.unsqueeze(0).to(device))
    # fls = [truncflat]
    #prepared_pgraphs = prepare_pgraphs(fls, pred[0])
    #bestpath = dp_pgraph(prepared_pgraphs[0], scofunct)
    #best = xlm_tok.decode(bestpath)
    blist = []
    usednodes = []
    for decround in range(numruns):
        pnodes = prepare_nodes([flnodes[:512-(posadd)]], pred[0], posadd)
        dpath, beplist, besclist = dynamic_path(pnodes[0], scofunct, posadd, usednodes)
        usednodes.extend([dp.token_idx for dp in dpath])
        blist.append(xlm_tok.decode([dp.token_idx for dp in dpath]))
    if verbose:
        print("SRC - "+graph['input'])
        print("PRED - "+blist[0])
        print("REF - "+graph['ref'])
    # verbose return for debugging
    if extra:
        return blist , flattened, pnodes, mask, sents, posids, pred, posadd, flnodes, dpath, beplist, besclist, totnodes
    return blist

"""
if __name__=="main":
    modtmp = XLMCometEmbeds(drop_rate=0.1)
    modtmp.load_state_dict(torch.load("./torchsaved/maskedcont4.pt"))
    modtmp.eval()
    base = "frtest_reversed/"
"""


# set scores computed for each token by the model
def set_pgscores(pgraphs, scores):

    for p in range(len(pgraphs)):
        pgraph = pgraphs[p]
        for i in range(min(len(pgraph),512)):
            pgraph[i]['score'] = scores[p][i]
            if pgraph[i]['token_idx']>0:
                if pgraph[i]['score']==0:
                    print(i)
                    print(p)
    return pgraphs

# topological sort the graphs, make sure that nodes that are next always come next in the list
def topo_sort_pgraph(pgraph):
    cp = [p for p in pgraph]
    tmpgraph = {}
    for c in cp:
        tmpgraph[c['id']]=c
    res = []
    visited = []
    # reverse ordering
    topo_sort_recurse(cp[0]['id'], res, [], tmpgraph)

    res.reverse()
    return res
        
def topo_sort_recurse(curid, toplist, visited, graph):
    if curid in visited:
        return 
    # for stuff in truncated part of graph TODO
    if curid not in graph.keys():
        return
    node = graph[curid]
    visited.append(curid)
    for nid in node['nexts']:
        topo_sort_recurse(nid, toplist, visited, graph)
    # once done, add to the beginnings
    toplist.insert(0, node)

def prepare_pgraphs(pgraphs, scores):
    res = []
    # make a deep copy of processed graphs
    for p in pgraphs:
        res.append([x for x in p])
    # set scores for stuff
    set_pgscores(res, scores)
    # do topological sorting
    newres = []
    for r in res:
        newres.append(topo_sort_pgraph(r))
    return newres

def default_scofunct (node):
    return node['score']

# given a list of sub-scores (topological flattening of the graph), use dp to get the highest scoring path
# idlist has the corresponding graph ids for 
# would need to do a sort on pgrapaps that makes sure that no next node is before in the linear ordering
# reverse since we're using nexts
# TODO simplify code to not need so many data structures
def dp_best_path(pgraphs, graph, sco_funct):
    bplist = []
    bsco_list =[]
    idlist = get_idlist(pgraphs)
    for i in range(len(idlist)):
        bpath = []
        cur = pgraphs[i]
            
        # get the highest prev from ahead to use
        mval = -10
        maxnext = None
        for n in cur['nexts']:
            try:
                if graph[n]['bestsco']>mval:
                    mval = graph[n]['bestsco']
                    maxnext = graph[n]
            except:
                ""
        cursco = sco_funct(cur)
        # add in scores / path from that prev
        if maxnext==None:
            bpath.append(i)
            bplist.append(bpath)
            bsco_list.append(cursco)
            # check if this is how things work in python
            graph[cur['id']]['bestsco'] = cursco
            graph[cur['id']]['plist'] = bpath
            continue
        bpath.extend(maxnext['plist']+[i])
        bplist.append(bpath)
        bsco_list.append(cursco+mval)
        graph[cur['id']]['bestsco'] = cursco+mval
        graph[cur['id']]['plist'] = bpath
        #print(bpath)
    return bplist[-1], bsco_list[-1]

def get_idlist(pgraph):
    return [p['id'] for p in pgraph]

def dp_pgraph(pgraph, scofunct):
    graph = {}
    for p in pgraph:
        # TODO check if scores are negative number compatible
        p['bestsco'] = 0
        p['plist'] = []
        graph[p['id']] = p
    bestpath, bsco = dp_best_path(pgraph, graph, scofunct)
    print(bsco)
    bestpath.reverse()
    return [pgraph[x]['token_idx'] for x in bestpath]
