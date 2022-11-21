import torch
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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
        # TODO made stuff deep copies, was there bad logic there?
        self.nextlist = [n for n in oldnode.nextlist]
        self.next_scores = [n for n in oldnode.next_scores]
        self.next_ids = [n for n in oldnode.next_ids]
        self.prevs = []
        self.detoks = []
        self.pos = -1
        self.canvpos = 1000
        self.dppos = -1
        self.score = 0
        # 0 unvisited, 1 in progress, 2 is visited
        self.visitval = 0
        # TODO may be something weird happening
        if hasattr(oldnode, "canvpos"):
            self.prevs = [p for p in oldnode.prevs]
            self.pos = oldnode.pos
            self.detoks = [d for d in oldnode.detoks]
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
        if fun is not consolidate_node:
            cur.visitval = 2
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


flat = []
def get_flat_lattice(gr):
    global flat
    flat = []
    wordgraph = combine_nodes(gr)
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
    cp = [f for f in flat]
    return cp

current_flat = []
allflats = []
def get_derecomb_lattice(gr):
    global current_flat
    global allflats
    global flat
    current_flat = []
    allflats = []
    flat = []
    # TODO compression at the beginning can cut down node count probably
    wordgraph = combine_nodes(gr)
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
        
    greedy_traverse(wordgraph, add_derecomb)
    # get the last canvas, unless perfect fit
    if len(current_flat)>0:
        allflats.append(current_flat)
    # print("Greedy traversal - ", len(flat))
    af = []
    for flattmp in allflats:
        af.append([f for f in flattmp])
    # return a copy of the whole flat thing
    return af

# Description of algorithm for de-recomb + flatten
#   - Decode greedily
#   - Whenever we hit a node w recomb
#       - Make a new node for each prev, ensure only 1 prev link / node
#       - This will push to the end in O(final number of nodes), not sure if that's good
#   - Keep track of space left in canvas (update after each ending)
#       - Also store nodes not added yet
#   - At end time, if enough space, then just extend canvas (clear stored nodes)
#       - Otherwise, track back (we ensure each only has 1 path back w/ above algo)
#       - Get canonical context / add into new canv, continue algo from there
# TODO make a general method for handling node manipulation or smth, to avoid bugs
def add_derecomb(node, grph):
    global current_flat
    global allflats
    global flat
    add_to_flat(node, grph)
    # we have recomb, by algo we can assume that this is the first time hitting node
    if len(node.prevs)>1:
        check = 0
        # will the last node just be the one that led to current?
        pkeep = []
        for prev in node.prevs:
            # this zhould only happen once
            if prev.visitval==2:
                pkeep.append(prev)
                continue 
            # make copy node, connect / disconnect as necessary
            tmpnode = DLReverseNode(node)
            tmpnode.uid = tmpnode.uid+prev.uid
            # only 1 prev
            tmpnode.prevs = [prev]
            tmpnode.canvpos = 1000
            # add the forward connection
            for n in node.nextlist:
                n.prevs.append(tmpnode)
            # ensure that backward connection is corrected
            if node in prev.nextlist:
                prev.nextlist.remove(node)
            if node.uid in prev.next_ids:
                prev.next_ids.remove(node.uid)
            prev.nextlist.append(tmpnode)
            prev.next_ids.append(tmpnode.uid)
            # TODO not sure if this is needed
            grph[tmpnode.uid] = tmpnode
            check+=1
        node.prevs = pkeep
        # TODO come back and fix this
        # assert len(pkeep)==1

    # check if we can add into full lattice or not
    if len(node.nextlist)==0:
        # TODO transfer limit based on src length
        if len(flat)+len(current_flat) > CANVLIM:
            allflats.append(current_flat)
            while len(flat[0].prevs)>0:
                assert len(flat[0].prevs)==1
                flat.insert(0, flat[0].prevs[0])   
            # track back to get the path leading to current node
            current_flat = [f for f in flat]
            flat = []
        else:
            # we fit into the canvas, can just add in
            current_flat.extend(flat)
            flat = []

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
    node.canvpos = len(flat) - 2 + len(detok_tmp)

# take a word node, convert it back to tokenized nodes
def split_dl_node(node):
    if len(node.detoks)==1:
        node.token_idx = node.detoks[0]
        node.token_str = detok.decode(node.detoks[0])
        return [node]
    res = []
    if len(node.detoks)==0:
        # special case with no token for some reason, TODO may need to examine
        print("empty token")
        for prev in node.prevs:
            if node in prev.nextlist:
                prev.nextlist.remove(node)
            if node.uid in prev.next_ids:
                prev.next_ids.remove(node.uid)
            prev.nextlist.extend(node.nextlist)
            prev.next_ids.extend(node.next_ids)
        for nextn in node.nextlist:
            if node in nextn.prevs:
                nextn.prevs.remove(node)
            nextn.prevs.extend(node.prevs)
        return []
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
    #print("original nodes - ", len(gr.keys()))
    tmplist = gr['root'].nextlist[0].nextlist
    tmpids = gr['root'].nextlist[0].next_ids
    gr['root'].nextlist = tmplist
    gr['root'].next_ids = tmpids
    flatlat = get_flat_lattice(gr)
    res = []
    #print("flatlat - ", len(flatlat))

    for f in flatlat:
        res.extend(split_dl_node(f))
    #print("final detokd - ", len(res))
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
            
    for g in goneprevs:
        node.prevs.remove(g)
    if len(node.prevs)==0:
        throw_garbage(node, grph)

def extend_unique(lis1, lis2):
    for l in lis2:
        if l not in lis1:
            lis1.append(l)
        
def update_removed(oldnode, newnode):
    for p in oldnode.prevs:
        if oldnode in p.nextlist:
            p.nextlist.remove(oldnode)
        if oldnode.uid in p.next_ids:
            p.next_ids.remove(oldnode.uid)
        p.nextlist.append(newnode)
        p.next_ids.append(newnode.uid)

    for n in oldnode.nextlist:
        if oldnode in n.prevs:
            n.prevs.remove(oldnode)
        n.prevs.append(newnode)

# takes in input string
def get_dictlist(grphinp, addnodes=False, compress=True):
    if type(grphinp)==str:
        gra = pickle.load(open(grphinp, 'rb'))
    else:
        gra = grphinp
    fllat = tokenize_flat_lattice(gra)
    flres = []
    if compress:
        tmpgraph = {}
        for tmpnode in fllat:
            tmpid = str(tmpnode.token_idx)+str(tmpnode.pos)
            if tmpid in tmpgraph.keys():
                # this is a duplicate node, compress with old one
                extend_unique(tmpgraph[tmpid].nextlist, tmpnode.nextlist)
                extend_unique(tmpgraph[tmpid].next_ids, tmpnode.next_ids)
                extend_unique(tmpgraph[tmpid].prevs, tmpnode.prevs)
                update_removed(tmpnode, tmpgraph[tmpid])
            else:
                flres.append(tmpnode)
                tmpgraph[tmpid] = tmpnode
    else:
        flres = fllat
    tdicts = []
    for f in flres:
        tdicts.append({
            'token_idx': f.token_idx,
            "token_str": f.token_str,
            'pos': f.pos, 
            'id': f.uid,
            'nexts': [fn.uid for fn in f.nextlist], 
            # TODO need to verify what this score actually is first, might already be nll
            'prob': f.prob, 
            #'score': math.log(f.prob), 
        })
    if addnodes:
        assert len(tdicts)==len(flres)
        return tdicts, flres
    return tdicts

import pickle
def test_flatten(ind, basedir):
    g = pickle.load(open(basedir+str(ind), 'rb'))
    #if g['input'] in old['src']:
    #    return None, None, None
    #try:
    return get_dictlist(g, True)

# Description of algorithm for de-recomb + flatten
#   - Decode greedily
#   - Whenever we hit a node w recomb
#       - Make a new node for each prev, ensure only 1 prev link / node
#       - This will push to the end in O(final number of nodes), not sure if that's good
#   - Keep track of space left in canvas (update after each ending)
#       - Also store nodes not added yet
#   - At end time, if enough space, then just extend canvas (clear stored nodes)
#       - Otherwise, track back (we ensure each only has 1 path back w/ above algo)
#       - Get canonical context / add into new canv, continue algo from there
# test code 


    
def construct_toy_recomb():
    sents = ["I am nice today", "He is nice today"]
    toks = toker(sents).input_ids
    nodes = [[], []]
    # TODO make sure lengths are the same
    grphtmp = {}
    for j in range(len(toks[0])):
        nodes[0].append(ReverseNode(None, {
            'uid':str(toks[0][j]),
            'prob':1,
            'token_idx':toks[0][j],
            'token_str':toker.decode(toks[0][j])
        }))
        if toks[1][j] == toks[0][j]:
            nodes[1].append(nodes[0][-1])
        else:
            nodes[1].append(ReverseNode(None, {
                'uid':str(toks[1][j]),
                'prob':1,
                'token_idx':toks[1][j],
                'token_str':toker.decode(toks[1][j])
            }))
        grphtmp[nodes[0][-1].uid] = nodes[0][-1]
        grphtmp[nodes[1][-1].uid] = nodes[1][-1]
    for i in range(len(toks)):
        for j in range(len(toks[0])-1):
            if nodes[i][j+1].uid not in nodes[i][j].next_ids:
                nodes[i][j].nextlist.append(nodes[i][j+1])
                nodes[i][j].next_ids.append(nodes[i][j+1].uid)
                nodes[i][j].next_scores.append(.5)
        nodes[i][len(toks[0])-1].nextlist = []
        nodes[i][len(toks[0])-1].next_ids = []
        nodes[i][len(toks[0])-1].next_scores = []
    grphtmp['root'] = nodes[0][0]
    return grphtmp

# TODO construct a toy test case with recomb
if __name__ == "__main__":
    g = pickle.load(open(base+str(0), 'rb'))
    #g = construct_toy_recomb()
    CANVLIM = 400
    
    aflats = get_derecomb_lattice(g)
    # TODO what if a new decoding has a new leadup path? Need to fill in missing stuff...
    print(len(aflats))

# TODO maybe do something about duplicate nexts / prevs