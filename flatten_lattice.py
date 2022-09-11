
from transformers import AutoTokenizer 
import pickle
import numpy as np
import os

GBASE = "./reverse_graphs/"
endebase = "mt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9/"
frenbase = "mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.902_0.0_0.9/"
bert_tok = AutoTokenizer.from_pretrained('bert-base-cased')
mbart_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
mbart_tok.src_lang = "en_XX"

# method that takes in # of stops
def get_processed_graph_data(lanbase, stop=-1, msplits=-1):
    # variable tracks during lattice decoding and stops after a certain point
    global max_splits
    global splits_hit
    max_splits = msplits
    base = GBASE+lanbase
    paths = os.listdir(base)
    result = []
    if stop==-1:
        stop = len(paths)
    # get flattened version of lattice for each lattice in directory
    for i in range(0, stop):
        splits_hit=0
        curgraph = load_graph(base+paths[i])
        # get rid of duplicate nodes TODO [happens surprisingly often?]
        result.append(remove_duplicates(flatten_lattice(curgraph)))
    return result

# flatten out lattice 
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
def greedy_flatten(tdicts, visited, node, pos, prev_cont, added_ids, branch_start=None):
    global splits_hit
    if node.uid in visited:
        return
    if node.token_idx==2 or node.token_idx==250004:
        npos = pos
    else:
        node.pos = pos
        
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
                # TODO add logic that tracks scores / next nodes
                otdlen = len(tdicts)
                for bind in range(0, len(bert_toks)):
                    b = bert_toks[bind]
                    if b==101 or b==102:
                        continue
                    nid = str(b)+" "+str(curpos)
                    # if we're at the start, add this node to next of branch node
                    if len(tdicts)==otdlen and branch_start is not None:
                        branch_start['nexts'].append(nid)

                    if bind<len(bert_toks)-1:
                        tdicts.append({
                            'token_idx':b,
                            'pos':curpos, 
                            'id': nid,
                            'nexts': [str(bert_toks[bind+1])+" "+str(curpos+1)], 
                            'score': 0
                        })
                    else:
                        tdicts.append({
                            'token_idx':b,
                            'pos':curpos, 
                            'id': str(b)+" "+str(pos),
                            'nexts': [], 
                            'score': 0
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
    if branched and len(tdicts)>0:
        branch_start=tdicts[-1]
    # TODO check which direction we need to go from argsort
    for i in range(0, len(scosort)):
        greedy_flatten(tdicts, visited, node.nextlist[scosort[i]], npos, prev_cont, added_ids, branch_start)

# get token list that tracks back a single path from start
def find_backs(start, fgraph, covered):
    cur = start
    nid = fgraph[cur]['id']
    res = []
    while cur>=0:
        cur = cur-1
        if nid in fgraph[cur]['nexts']:
            covered.add(fgraph[cur]['id']+str(fgraph[cur]['nexts']))
            res = [fgraph[cur]['token_idx']] + res
            nid = fgraph[cur]['id']
    return res

# get token list that tracks canonical forward
def find_fronts(start, fgraph, covered):
    nid = fgraph[start]['nexts'][-1]
    res = []
    for i in range(start, len(fgraph)):
        
        if nid == fgraph[i]['id']:
            covered.add(nid+str(fgraph[i]['nexts']))
            #print(nid)
            res = res + [fgraph[i]['token_idx']]
            nextmp = fgraph[i]['nexts']
            if len(nextmp)==0:
                return res
            elif len(nextmp)>1:
                nid = nextmp[1]
            else:
                nid = nextmp[0]
        
    return res

# will get a smaller numebr of paths that cover all the tokens in the graph, should make gold-generation quicker / hopefully less buggy
def get_cover_paths(fgraph):
    covered = set()
    res = []
    for i in range(len(fgraph)):
        f = fgraph[i]
        if f['id']+str(f['nexts']) not in covered:
            tmp = find_backs(i, fgraph, covered)+[f['token_idx']]+find_fronts(i, fgraph, covered)
            res.append(bert_tok.decode(tmp))
            covered.add(f['id']+str(f['nexts']))
            
    return res, covered

# get rid of exact duplicate nodes in lattice canvas (do sanity check to make sure nodes aren't being excluded somehow)
def remove_duplicates(fgraph):
    ngraph = []
    idlist = set()
    for f in fgraph:
        if f['id']+str(f['nexts']) in idlist:
            continue
        idlist.add(f['id']+str(f['nexts']))
        ngraph.append(f)
    return ngraph
    
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
    
def find_tdict_node(tdictions, idval):
    for t in tdictions:
        if t['id']==idval:
            return t

def greedy_path(flat):
    prev = -1
    res = []
    for f in flat:
        if f.pos>prev:
            res.append(f)
            prev = f.pos
    return res
