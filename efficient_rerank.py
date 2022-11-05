from asyncio import new_event_loop
import flatten_lattice as fl
import torch
from encoding_utils import *
from new_mask_utils import get_causal_mask

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from new_flatten_lattice import get_dictlist

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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

def run_pipeline(graph, model, scofunct, extra=False):
    #flatold = fl.flatten_lattice(graph)
    flattened, flnodes = get_dictlist(graph, True)
    ppinput = prepend_input(flattened, graph['input'])
    flattened = ppinput[0]
    # covered = fl.get_cover_paths(flattened)

    posadd = ppinput[1]
    mask = get_causal_mask(flnodes, posadd)
    sents, posids = create_inputs([flattened])
    with torch.no_grad():
        pred = model(sents, posids, mask.unsqueeze(0).to(device))
    fls = [flattened]
    prepared_pgraphs = prepare_pgraphs(fls, pred[0])
    bestpath = dp_pgraph(prepared_pgraphs[0], scofunct)
    best = xlm_tok.decode(bestpath)
    print("SRC - "+graph['input'])
    print("PRED - "+best)
    print("REF - "+graph['ref'])
    # verbose return for debugging
    if extra:
        return best , flattened, prepared_pgraphs, mask, sents, posids, pred, posadd, flnodes
    return best


    
