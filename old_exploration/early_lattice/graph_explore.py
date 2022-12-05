from distutils import filelist
import pickle
from tabnanny import check
from unittest import result
from weakref import finalize
import pandas as pd
import os 
from src.recom_search.evaluation.analysis import derive_path, viz_result

MAX_SUBLEN = 10
all_shared_paths = []
explored = []
BASE = './output/data/sum_xsum_astar_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9'

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
        s = p.token_str+s
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

def get_all_path_recombs(fname):
    global all_shared_paths
    global cur_end
    graph = get_graph(BASE+fname)
    for i in range(0, len(graph.ends)):
        cur_end = graph.ends[i]
        get_reconv_paths(graph.ends[i], "")
        print(len(all_shared_paths))
    asp = all_shared_paths
    all_shared_paths = []
    return asp

def print_shared_path (sp):
    print("-------")
    print(sp['s1'])
    print(sp['s2'])

# leave flip pruning to later processing / analysis
def prune_paths(paths):
    res = []
    [res.append(x) for x in paths if x not in res]
    result = []
    # remove cases when the words are the same, check if this is ok
    for r in res:
        if r['p1'] == r['p2']:
            continue
        result.append(r)
    return result

def get_split_meta(path):
    tokids, probs, scores = [], [], []
    for p in path:
        tokids.append(p.uid)
        probs.append(p.prob)
        scores.append(p.score)
    return tokids, probs, scores

# since jupyter can't handle code from this dir, we'll generate a ton of metadata 
# that can be quick / easy to load, may need to update whenever we do more graph 
# structure analysis
def finalize_metadata (sp):
    res = []
    CONT = 5
    for s in sp: 
        try:
            s['context'] = str(get_node(s['p1'][-1], 0))
        except:
            s['context'] = ""
        #s['afterstr'] = pathlist_str(get_after(s['afterstr'], s['p1'][0])[:-1])
        t1, p1, s1 = get_split_meta(s['p1'])
        s['ids1'] = t1
        s['probs1'] = p1
        s['scores1'] = s1
        t2, p2, s2 = get_split_meta(s['p2'])
        s['ids2'] = t2
        s['probs2'] = p2
        s['scores2'] = s2
        s['p1'] = ""
        s['p2'] = ""
        res.append(s)
    return res

def get_ids_and_files():
    paths = os.listdir(BASE)
    result = {}
    for p in paths:
        tmpid = ""
        tmpid = p[63:]
        tmpid = tmpid.split("_")[0]
        result[tmpid] = p
    return result

def process_save_all_graphs():
    filedict = get_ids_and_files()
    result = {}
    for f in filedict.keys():
        fnam = filedict[f]
        sharedpaths = get_all_path_recombs(fnam)
        sharedpaths = prune_paths(sharedpaths)
        print("ID: "+f)
        print(len(sharedpaths))
        result[f] = finalize_metadata(sharedpaths)
        
    return result

def run_save_graphs():
    allgraphs = process_save_all_graphs()
    import json
    with open('subgraph_meta/data.json', 'w', encoding='utf-8') as f:
        json.dump(allgraphs, f, ensure_ascii=False, indent=4)

def print_all_sharedpaths(sps):
    for s in sps:
        print_shared_path(s)

def save_ind_graphs (ind):
    filedict = get_ids_and_files()
    result = {}
    count = 0
    flist = list(filedict.keys())
    f = flist[ind]
    fnam = filedict[f]
    sharedpaths = get_all_path_recombs(fnam)
    sharedpaths = prune_paths(sharedpaths)
    print("ID: "+f)
    print(len(sharedpaths))
    result[f] = finalize_metadata(sharedpaths)
    count+=1
    return result

run_save_graphs()
#save_ind_graphs(2)