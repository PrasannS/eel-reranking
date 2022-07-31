from distutils import filelist
import pickle
from tabnanny import check
from unittest import result
from weakref import finalize
from matplotlib.cbook import get_sample_data
import pandas as pd
import os 
from src.recom_search.evaluation.analysis import derive_path, viz_result
from transformers import AutoTokenizer
import argparse

def process_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-savename', type=str)
    parser.add_argument('-device', type=str, default='cuda:2')
    parser.add_argument('-exploded', type=bool, default=False)

    args = parser.parse_args()
    return args


all_shared_paths = []
explored = []
# using for french stuff that worked
BASE = './output_fr_en_1/data/sum_xsum_astar_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9/'

def load_save_data(fname):
    f = open(fname, 'rb')
    graph = pickle.load(f)
    pickle.dump(graph, open('graphtmp.pkl','wb'))

def get_graph(fname):
    f = open(fname, 'rb')
    return pickle.load(f)
#load_open_data('./output_v1/data/sum_xsum_astar_16_15_False_0.4_True_False_4_5_zip_0.75_0.0_0.9/sum_xsum_astar_16_15_False_0.4_True_False_4_5_zip_0.75_0.0_0.9_28360838_The-law-al.pkl')
# method, returns backwards list with most likely path to a given node
def get_top1_path(node):
    node_list = []
    node_list.append(node)
    cur = node
    while len(cur.prev) > 0:
        cur = cur.hash.retrieve_node(cur.prev[0])
        node_list.append(cur)
        #print(cur.token_str)
    return node_list

cur_end = None

def get_after(end, node):
    end = node.hash.retrieve_node(end)
    afterlist = []
    get_after_helper(afterlist, end, node)
    return afterlist

def get_after_helper(alist, cur, target):
    if cur.uid==target.uid:
        return True
    for p in cur.prev:
        tmp = cur.hash.retrieve_node(p)
        alist.append(tmp)
        if get_after_helper(alist, tmp, target):
            return True
        alist.pop()
    return False
    
# get list from end to start of top uids of a given path going back to start
def get_top1_uids(node):
    node_list = []
    node_list.append(node.uid)
    cur = node
    while len(cur.prev) > 0:
        cur = cur.hash.retrieve_node(cur.prev[0])
        node_list.append(cur.uid)
        #print(cur.token_str)
    return node_list

# check if 2 nodes started from the same path within the last 10 top nodes, if so, 
# return an object with a list of the sub-paths to compare
def check_reconv(n1, n2):
    l1 = get_top1_uids(n1)
    l2 = get_top1_uids(n2)
    ind1 = -1
    ind2 = -1
    for i in range(0, MAX_SUBLEN):
        if l1[i] in l2:
            ind1 = i
            ind2 = l2.index(l1[i])
            break
    # edge case of 0
    if ind1>0:
        sharedpath = {}
        # todo make a more efficient top_path method
        sharedpath['p1'] = get_top1_path(n1)[:ind1]
        sharedpath['p2'] = get_top1_path(n2)[:ind2]
        return sharedpath
    return None

def get_node (node, ind):
    return node.hash.retrieve_node(node.prev[ind])

def pathlist_str(pathl):
    s = ""
    for p in pathl:
        s = p.token_str+" "+s
    return s
    
# get up to n words that come before sequence
def get_context(n, pl):
    node = pl[-1]
    i = 0
    contlist = []
    while i < n and len(node.prev)>0:
        node = get_node(node, 0)
        contlist.append(node)
        i+=1
    return contlist

def clean_path (path):
    i = 0
    while i<len(path)-1:
        if path[i].token_str==path[i+1].token_str:
            del path[i]
        else:
            i+=1
    return path
            

# might need to be recursive, generate global list of reconverging paths
def get_reconv_paths(node, afterstr):
    global all_shared_paths
    global explored
    if node.uid in explored:
        return
    explored.append(node.uid)
    cur = node
    is_split = False
    tmp = afterstr
    while len(cur.prev) > 0:
        tmp = cur.token_str+tmp
        if len(cur.prev) > 1:
            # check combinations and paths of split nodes
            for i in range(0, len(cur.prev)):
                for j in range(0, i):
                    if i == j:
                        continue
                    path = check_reconv(get_node(cur, i), get_node(cur, j))
                    if path != None:
                        #path['p1'] = clean_path(path['p1'])
                        #path['p2'] = clean_path(path['p2'])
                        path['s1'] = pathlist_str(path['p1'])
                        path['s2'] = pathlist_str(path['p2'])
                        path['afterstr'] = tmp+""
                        all_shared_paths.append(path)
            is_split = True
            break
        else:
            explored.append(node.uid)
            cur = get_node(cur, 0)
    if is_split:
        # might lead to repeats, but recursively push method to future nodes
        for i in range(0, len(cur.prev)):
            get_reconv_paths(get_node(cur, i), tmp+"")

def get_path_sample(fname):
    global all_shared_paths
    global cur_end
    graph = get_graph(BASE+fname)
    scores =  []
    cands = []
    ref = graph.reference
    src = graph.document
    for e in graph.ends:
        scores.append(e.get_score_avg())
        cands.append(get_node_str(e))
    return scores, cands, ref, src

def get_node_str(node):
    c = get_context(50, [node])
    toks = [con.token_idx for con in c]
    toks.reverse()
    toks.append(node.token_idx)
    s = tokenizer.batch_decode([toks])[0]
    s = s.replace("</s>", "")
    s = s.replace("de_DE", "")
    return s.strip()

def get_plist_str(plist):
    toks = [con.token_idx for con in plist]
    toks.reverse()
    s = tokenizer.batch_decode([toks])[0]
    s = s.replace("</s>", "")
    s = s.replace("de_DE", "")
    return s.strip()

def get_plist_sco(plist):
    tot = 0
    for p in plist:
        tot+=p.score
    return tot/len(plist)

def find_paths(root):
    #print(root.token_str)
    if len(root.prev) == 0:
        yield [root]

    for c in range(0, len(root.prev)):
        child = get_node(root, c)
        for path in find_paths(child):
            yield [root] + path
        
# TODO, wait, does this just mean that recombination doesn't do anything
# use candidates from this as a theoretical bound on what lattices can do
def get_all_possible_candidates(fname):
    graph = get_graph(BASE+fname)
    scores =  []
    cands = []
    ref = graph.reference
    src = graph.document
    fullplist = []
    for e in graph.ends:
        fullplist.extend(list(find_paths(e)))
    for plist in fullplist:
        scores.append(get_plist_sco(plist))
        cands.append(get_plist_str(plist))
    
    # TODO some kind of filtration that prevents super similar or bad stuff from being used
    return scores, cands, ref, src

osetnum = 50

def get_ids_and_files():
    paths = os.listdir(BASE)
    result = {}
    for p in paths:
        tmpid = ""
        tmpid = p[osetnum:]
        #tmpid = tmpid.split("_")[0]
        result[tmpid] = p
    assert len(paths) == len(result.keys())
    return result

def process_save_all_graphs(explode):
    filedict = get_ids_and_files()
    results = []
    for f in filedict.keys():
        fnam = filedict[f]
        # scores, candidates, reference, input doc
        # s, c, r, d = get_path_sample(fnam)
        # heavy duty version
        if explode:
            s, c, r, d = get_all_possible_candidates(fnam)
        else:
            s, c, r, d = get_path_sample(fnam)
        cand_sorted = [i for _,i in sorted(zip(s,c))]
        sco_sorted = sorted(s)
        cand_sorted.reverse()
        sco_sorted.reverse()
        tmp = {}
        tmp['scores'] = sco_sorted
        tmp['cands'] = cand_sorted
        tmp['inp'] = d
        tmp['ref'] = r
        results.append(tmp)
        
    return results

if __name__ == "__main__":
    args = process_args()

    if args.dataset=='en_de':
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

    #run_save_graphs()
    latticecandjson = process_save_all_graphs()
    print(latticecandjson[0])

    import json
    tmpfile = open("./candoutputs/"+args.savename+".jsonl", "w")
    for l in latticecandjson:
        tmpfile.write(json.dumps(l))
        tmpfile.write('\n')
    tmpfile.close()