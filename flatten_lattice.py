import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer #, AutoModel
import pickle
import numpy as np
import os
from src.recom_search.model.beam_node_reverse import ReverseNode

GBASE = "./reverse_graphs/"
endebase = "mt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9/"
frenbase = "mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.902_0.0_0.9/"
bert_tok = AutoTokenizer.from_pretrained('bert-base-cased')
mbart_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
mbart_tok.src_lang = "en_XX"

def load_graph(fname):
    return pickle.load(open(fname,'rb'))

def get_toklist(revnodes):
    res = []
    for r in revnodes:
        if type(r) is dict:
            res.append(r['token_idx'])
        else:
            res.append(r.token_idx)
    return res

def flatten_lattice(graph):
    tokdicts = []
    visited = []
    prev_contig = []
    greedy_flatten(tokdicts, visited, graph['root'], 0, prev_contig, set())
    #greedy_flat_old(tokdicts, visited, graph['root'], 0)
    return tokdicts
    
max_splits = -1
splits_hit = 0
# flattens graph by position, ignores </s> and en_XX tokens for greater BERT compatibility
# TODO set up to use mbart tokenization
def greedy_flatten(tdicts, visited, node, pos, prev_cont, added_ids):
    global splits_hit
    if node.uid in visited:
        #print("cycle here")
        return
    if node.token_idx==2 or node.token_idx==250004:
        npos = pos
    else:
        node.pos = pos
        # tdicts.append(node)
        visited.append(node.uid)
        npos = pos+1
        s = node.token_str
        prev_cont.append(node)
    
    olen = len(tdicts)
    # we're hitting a branch or an ending, update to bert tokenization and add to visited
    # should be ok to do this since branching / merging only happens at word boundaries (presumably)
    branched = (len(node.next_scores)>1)
    end = (len(node.next_scores)==0)
    merge = end==False and node.nextlist[0].uid in visited
    if branched or merge or end:
        #print(branched, " ", merge, " ", end)
        if len(prev_cont)>0:
            errorflag = False
            
            prev_update = []
            for p in prev_cont:
                if p.uid in added_ids:
                    continue
                else:
                    prev_update.append(p)
                    added_ids.add(p.uid)
            
            if len(prev_update)>0:
                toktmp = get_toklist(prev_update)
                for i in range(1, len(prev_update)):
                    if prev_update[-(i+1)].pos>=prev_update[-i].pos:
                        errorflag = True
                        break
                decstr = mbart_tok.decode(toktmp)
                if errorflag:
                    #print(decstr)
                    #print([p.pos for p in prev_update])
                    ""
                bert_toks = bert_tok(decstr).input_ids
                curpos = prev_update[0].pos
                # TODO add logic that tracks scores
                for b in bert_toks:
                    if b==101 or b==102:
                        continue
                    tdicts.append({
                        'token_idx':b,
                        'pos':curpos
                    })
                    curpos+=1
                if merge or end:
                    splits_hit+=1
            
    if len(tdicts)>olen:
        del prev_cont
        prev_cont = []
        
    # end things early if we want to limit paths
    if max_splits>=0 and splits_hit>=max_splits:
        return 
    
    scosort = list(np.argsort(node.next_scores))
    
    # TODO check which direction we need to go from argsort
    for i in range(0, len(scosort)):
        greedy_flatten(tdicts, visited, node.nextlist[scosort[i]], npos, prev_cont, added_ids)
        
def get_processed_graph_data(lanbase, stop=-1, msplits=-1):
    global max_splits
    global splits_hit
    max_splits = msplits
    base = GBASE+lanbase
    paths = os.listdir(base)
    #print(len(paths))
    result = []
    if stop==-1:
        stop = len(paths)
    for i in range(0, stop):
        splits_hit=0
        curgraph = load_graph(base+paths[i])
        result.append(flatten_lattice(curgraph))
    return result

def greedy_path(flat):
    prev = -1
    res = []
    for f in flat:
        if f.pos>prev:
            res.append(f)
            prev = f.pos
    return res

def find_paths(root):
    global nodeset
    #print(root.token_str)
    if len(root.nextlist) == 0:
        yield [root]

    scosort = list(np.argsort(root.next_scores))
    
    seen = []
    for s in scosort:
        child = root.nextlist[s]
        if child.uid in seen:
            continue
        nodeset.add(child.uid)
        #if len(seen)>1:
            #print("maybe not bug")
        seen.append(child.uid)
        for path in find_paths(child):
            yield [root] + path
            
def get_plist_sco(plist):
    res = []
    for p in plist:
        res.append(p.prob)
    return res

def get_plist_str(plist):
    res = []
    for p in plist:
        res.append(p.token_idx)
    val =  mbart_tok.decode(res)
    #print(val)
    return val

nodeset = set()
STOP = 1000
def get_all_possible_candidates(graph):
    global nodeset
    scores =  []
    cands = []
    fullplist = []
    generated = 0
    
    for p in find_paths(graph['root']):
        if generated == STOP:
            break
        fullplist.append(p)
        generated+=1
    #print("num nodes")
    #print(len(nodeset))
    nodeset = set()
    #fullplist = remove_dups(fullplist)
    #print("candidates")
    #print(len(fullplist))
    for plist in fullplist:
        #scores.append(get_plist_sco(plist))
        cands.append(get_plist_str(plist))
    
    # TODO some kind of filtration that prevents super similar or bad stuff from being used
    return cands
    
def get_allcands(lanbase, stop=-1, res=[]):
    base = GBASE+lanbase
    paths = os.listdir(base)
    #print(len(paths))
    if stop==-1:
        stop = len(paths)
    for i in range(0, stop):
        try:
            curgraph = load_graph(base+paths[i])
            res.append(get_all_possible_candidates(curgraph))
        except:
            #print("hit recursion limit")
            res.append([])
    return res
        #result.append(flatten_lattice(curgraph))
    #return result

#### OLD SANITY CHECKS ####
def check_encsame(flat):
    tlist = get_toklist(flat)
    decstr = mbart_tok.decode(get_toklist(flat))
    re_encoded = mbart_tok(decstr).input_ids
    #print(decstr)
    for i in range(0, len(tlist)):
        #print(mbart_tok.decode(tlist[i]), " ", mbart_tok.decode(re_encoded[i+1]))
        if tlist[i]==re_encoded[i+1]:
            continue
        #print(tlist[i])
        #print(re_encoded[i+1])
        #return False
    return True

def mbart_to_bert (flat):
    tlist = get_toklist(flat)
    decstr = mbart_tok.decode(get_toklist(flat))

def print_proctoks(revnodes):
    for rev in revnodes:
        print(rev.token_str, " - ", rev['pos'])

#print_proctoks(greedy_path(processedgraphs[2]))
#check_encsame(processedgraphs[4])