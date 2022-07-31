#rom distutils import filelist
import pickle
#from tabnanny import check
#from unittest import result
#from weakref import finalize
import os
from uuid import getnode
from factuality_datasets.eval_genoutputs_modified import getstuff, get_dae_output

#from src.recom_search.evaluation.analysis import derive_path, viz_result

MAX_SUBLEN = 10
all_shared_paths = []
explored = []
DIR_BASE = '/mnt/data1/prasann/latticegen/lattice-generation'
BASE = '/output/data/sum_xsum_astar_16_15_False_0.4_True_False_4_5_zip_0.75_0.0_0.9/'

testname = "sum_xsum_astar_16_15_False_0.4_True_False_4_5_zip_0.75_0.0_0.9_40726537_The-Englis.pkl"

tok, mod, nlp = getstuff()
## START GRAPH PROCESSING HELPERS
def load_save_data(fname):
    f = open(DIR_BASE+fname, 'rb')
    graph = pickle.load(f)
    pickle.dump(graph, open('graphtmp.pkl','wb'))

def get_graph(fname):
    f = open(DIR_BASE+fname, 'rb')
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
    return asp, graph

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
        sharedpaths, g = get_all_path_recombs(fnam)
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
    sharedpaths, g = get_all_path_recombs(fnam)
    sharedpaths = prune_paths(sharedpaths)
    print("ID: "+f)
    print(len(sharedpaths))
    result[f] = finalize_metadata(sharedpaths)
    count+=1
    return result

## END HELPERS FOR GRAPH_PROCESSING

# TODO factor in DAE dynamically into pipeline

def get_path_fullsums(sharedpath):
    try:
        summ1 = str(sharedpath['p1'][0]).replace("</s>", "").replace("-", "")
        summ2 = str(sharedpath['p2'][0]).replace("</s>", "").replace("-", "")
    except:
        return "the", "the"
    return summ1+sharedpath['afterstr'], summ2+sharedpath['afterstr']

pruneexplored = []

def prune_graph_end(end, prunelist):
    global pruneexplored
    if end.uid in pruneexplored:
        return
    pruneexplored.append(end.uid)
    print(end.uid)
    cur = end
    if len(cur.prev)>0:
        if len(cur.prev)>1:
            cur.prev = [x for x in cur.prev if not (x in prunelist)]
        for c in range(0, len(cur.prev)):
            prune_graph_end(get_node(cur, c), prunelist)

def prune_graph_from_list(prunelist, graph):
    global cur_end
    prunelist = set(prunelist)
    for i in range(0, len(graph.ends)):
        print(cur_end.uid)
        cur_end = graph.ends[i]
        prune_graph_end(graph.ends[i], prunelist)
            
    return graph
    

def hallucinated_token(sp, doc):
    # fs1, fs2 = get_path_fullsums(a)
    for t in sp['p1']:
        if t.token_str not in doc:
            return 1
    for t in sp['p2']:
        if t.token_str not in doc:
            return 2
    return 0

#given shared path and source document, figure out whether dae says to prune one or the other
def dae_check (sp, doc):
    fs1, fs2 = get_path_fullsums(sp)
    dae1 = get_dae_output(doc,fs1, tok, mod, nlp, None)
    dae2 = get_dae_output(doc,fs2, tok, mod, nlp, None)

    pcnt1 = dae1['preds'].count(0)
    pcnt2 = dae2['preds'].count(0)

    if pcnt1==pcnt2: 
        return 0
    if pcnt2>pcnt1:
        return 2
    return 1

#TODO do something to be able to cut off an entire branch (set end in sharedpath smth)

# if similarity falls behind then prune, unusable in current state (may need another metric)
"""
CUTOFF = 0.6
def bertscore_source(sp, doc):
    fs1, fs2 = get_path_fullsums(sp)
    p1, r1, f1 = ""
"""


# Method that 
# - goes through graph
# - looks at places where there's divergence
# - checks out full strings for things that are diverging
# - evaluates on some sort of metric
# - prunes branches which don't fit given metric
# - save graph that has gone through the process
# TODO make metric a generic method
def prune_graph_on_metric (metric, fname):
    sharedpaths, graph = get_all_path_recombs(fname)
    asp = prune_paths(sharedpaths)
    prunelist = []
    docum = graph.document
    
    for a in asp:
        num = metric(a, docum)
        # TODO check some condition, prune some path based on it
        if num==1:
            prunelist.append(a['p1'][0].uid)
        elif num==2:
            prunelist.append(a['p2'][0].uid)
    print(prunelist)
    global pruneexplored
    pruneexplored = []
    fingraph = prune_graph_from_list(prunelist, graph)
    # TODO save graph somewhere
    # TODO save data of pruned stuff

prune_graph_on_metric(dae_check, testname)
print("done")
