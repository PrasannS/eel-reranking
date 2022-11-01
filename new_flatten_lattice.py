import torch
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from transformers import AutoTokenizer
import pickle
import pandas as pd
import os
from src.recom_search.model.beam_node_reverse import ReverseNode
import numpy as np
import math 

class DLReverseNode():
    def __init__(self, oldnode):
        self.uid = oldnode.uid
        self.prob = oldnode.prob
        self.token_idx = oldnode.token_idx
        self.token_str = oldnode.token_str
        self.nextlist = oldnode.nextlist
        self.next_scores = oldnode.next_scores
        self.next_ids = oldnode.next_ids
        self.prevs = []
        self.detoks = []
        self.pos = -1
        self.canvpos = 1000
        # TODO may be something weird happening
        if hasattr(oldnode, "canvpos"):
            self.prevs = oldnode.prevs
            self.pos = oldnode.pos
            self.detoks = oldnode.detoks
            self.canvpos = oldnode.canvpos
        
    def __str__(self):
        return self.token_str

base = "frtest_reversed/"
toker = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
detok = AutoTokenizer.from_pretrained("xlm-roberta-base")

# TODO later on just move this to the initial graph reversal
def get_dbl_graph(grph):
    newgrph = {}
    for g in grph.keys():
        if g=="input" or g== "ref" or g== "rootid":
            continue
        tmp = DLReverseNode(grph[g])
        newgrph[g] = tmp
    for gk in newgrph.keys():
        if gk=="input" or gk== "ref" or gk== "rootid":
            continue
        # this should handle everything having a previous list
        newgrph[gk].nextlist = None
        newgrph[gk].nextlist = [newgrph[idl] for idl in newgrph[gk].next_ids]
    for gk in newgrph.keys():
        if gk=="input" or gk== "ref" or gk== "rootid":
            continue
        for n in newgrph[gk].nextlist:
            n.prevs.append(newgrph[gk])
        newgrph[gk].token_idx = [newgrph[gk].token_idx]
    # TODO do something to update scores on word graph
    return newgrph

# greedily traverse graph, run function on each node
def greedy_traverse(gr, fun, norev=True):
    queue = []
    queue.append(gr['root'])
    visited = []
    while len(queue)>0:
        cur = queue.pop()
        fun(cur, gr)
        order = np.argsort([n.prob for n in cur.nextlist])
        # highest prob gets popped off first
        for o in order:
            if cur.uid not in visited:
                queue.append(cur.nextlist[o])
        visited.append(cur.uid)
                
# make so graph has word-only nodes, check tokenization at different node boundaries
# update pointers afterwards
def combine_nodes(gr):
    dblgrph = get_dbl_graph(gr)
    #print("Doubly Linked - ", len(dblgrph.keys()))
    greedy_traverse(dblgrph, consolidate_node)
    rmlist = []
    for d in dblgrph.keys():
        if d=="input" or d== "ref" or "root" in d:
            continue
        if len(dblgrph[d].prevs)==0:
            rmlist.append(d)
    for r in rmlist:
        del dblgrph[r]
    return dblgrph

# do this on every node greedily to get flattened lattice canvas
def add_to_flat(node, grph):
    global flat
    if node.canvpos<1000:
        return
    flat.append(node)
    # get detokenization
    detok_tmp = detok(node.token_str).input_ids[1:-1]
    # update pos on first greedy hit
    if len(node.prevs)==0:
        node.pos = -1 + len(detok_tmp)
    if node.pos==-1:
        node.pos = max([n.pos for n in node.prevs])+len(detok_tmp)
    node.detoks = detok_tmp
    node.canvpos = len(flat)-2+ len(detok_tmp)
    
flat = []
def get_flat_lattice(gr):
    global flat
    flat = []
    wordgraph = combine_nodes(gr)
    # print("Combined nodes - ", len(wordgraph))
    # clear out prevs
    for gk in wordgraph.keys():
        if gk=="input" or gk== "ref" or gk== "rootid":
            continue
        wordgraph[gk].prevs = []
    # reset prevs
    for gk in wordgraph.keys():
        if gk=="input" or gk== "ref" or gk== "rootid":
            continue
        for n in wordgraph[gk].nextlist:
            n.prevs.append(wordgraph[gk])
    greedy_traverse(wordgraph, add_to_flat)
    # print("Greedy traversal - ", len(flat))
    cp = [f for f in flat]
    return cp

# take a word node, convert it back to tokenized nodes
def split_dl_node(node):
    if len(node.detoks)==1:
        node.token_idx = node.detoks[0]
        node.token_str = detok.decode(node.detoks[0])
        return [node]
    res = []
    # update previous of nodes
    for i in range(len(node.detoks)):
        n = node.detoks[i]
        # make a copy of our base node
        tmp = None
        tmp = DLReverseNode(node)
        if i>0:
            tmp.prevs = [res[-1]]
            # only have probability on 1
            tmp.prob = 1
        tmp.uid = tmp.uid+str(i)
        tmp.pos = tmp.pos - (len(node.detoks)-i-1)
        tmp.canvpos = tmp.canvpos - (len(node.detoks)-i-1)
        tmp.token_idx = n
        tmp.token_str = detok.decode(n)
        res.append(tmp)
    # update connection from previous
    for prev in node.prevs:
        prev.nextlist.remove(node)
        prev.next_ids.remove(node.uid)
        prev.nextlist.append(res[0])
        prev.next_ids.append(res[0].uid)
    # update next connection of nodes
    # TODO not updating the other end
    for i in range(len(res)-1):
        if i<(len(node.detoks)-1):
            tmpids = [res[i+1].uid]
            res[i].next_ids = tmpids
            tmplist = [res[i+1]]
            res[i].nextlist = tmplist
    return res
        
def tokenize_flat_lattice(gr):
    # get rid of first token, usually en_XX for french
    print("original nodes - ", len(gr.keys()))
    tmplist = gr['root'].nextlist[0].nextlist
    tmpids = gr['root'].nextlist[0].next_ids
    gr['root'].nextlist = tmplist
    gr['root'].next_ids = tmpids
    flatlat = get_flat_lattice(gr)
    res = []
    #print("flatlat - ", len(flatlat))

    for f in flatlat:
        res.extend(split_dl_node(f))
    print("final detokd - ", len(res))
    return res
    # we need to go through and convert this into lattices compatible 
    # with the format further into the pipeline, need to tokenize again with BERT
    
# disconnect / throw away node
def throw_garbage(node, grph, lprevs=False):
    assert len(node.prevs)==0 or len(node.nextlist)==0
    for pre in node.prevs:
        if node in pre.nextlist:
            pre.nextlist.remove(node)
        if node.uid in pre.next_ids:
            pre.next_ids.remove(node.uid)

    if lprevs:
        for n in node.nextlist:
            n.prevs.remove(node)
    if node.uid not in grph.keys():
        #print("w")
        ""
    else:
        del grph[node.uid]
        
# assume that previous nodes are consolidated, 
def consolidate_node(node, grph):
    if node.uid not in grph.keys():
        return
    goneprevs = []
    # check relationship with all previous nodes
    for prev in node.prevs:
        # it's a word boundary, no changes
        comb = toker.decode(prev.token_idx+node.token_idx)
        if " " in comb or "</s>" in comb:
            continue
        else:
            # need to make new node, add necessary stuff
            #print(comb)
            tmp = DLReverseNode(node)
            tmp.token_str = comb
            tmp.token_idx = prev.token_idx+node.token_idx
            tmp.prob = prev.prob*node.prob
            tmp.prevs = []
            tmp.prevs.extend(prev.prevs)
            tmp.uid = prev.uid+node.uid
            
            # connect previous nodes
            for pre in tmp.prevs:
                pre.nextlist.append(tmp)
                pre.next_ids.append(tmp.uid)
            
            grph[tmp.uid] = tmp
            # cut off from others where necessary
            goneprevs.append(prev)
            
            if node in prev.nextlist:
                prev.nextlist.remove(node)
                if node.uid in prev.next_ids:
                    prev.next_ids.remove(node.uid)
            
            for t in tmp.nextlist:
                t.prevs.append(tmp)
            # prev now garbage, delete it
            if len(prev.nextlist)==0:
                throw_garbage(prev, grph, True)
            """
            if comb=="China’":
                print("nexts after", len(tmp.nextlist))
                print(tmp.uid)
                print(tmp.nextlist[0].prevs[1].uid)
            """
            
    for g in goneprevs:
        node.prevs.remove(g)
    if len(node.prevs)==0:
        throw_garbage(node, grph)
        
# takes in input string
def get_dictlist(grphinp):
    if type(grphinp)==str:
        gra = pickle.load(open(grphinp, 'rb'))
    else:
        gra = grphinp
    fllat = tokenize_flat_lattice(gra)
    tdicts = []
    for f in fllat:
        tdicts.append({
            'token_idx': f.token_idx,
            "token_str": f.token_str,
            'pos': f.pos, 
            'id': f.uid,
            'nexts': [fn.uid for fn in f.nextlist], 
            'score': math.log(f.prob)
        })
    return tdicts

