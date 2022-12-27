import pickle
import os 
from transformers import AutoTokenizer
import argparse

# TODO switch need to change default back for all other lattices
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

def load_save_data(fname):
    f = open(fname, 'rb')
    graph = pickle.load(f)
    pickle.dump(graph, open('graphtmp.pkl','wb'))

def get_graph(fname):
    f = open(fname, 'rb')
    return pickle.load(f)

def get_node (node, ind):
    if hasattr(node, "hash"):
        return node.hash.retrieve_node(node.prev[ind])
    else:
        return node.prev[ind]
    
# get up to n words that come before sequence
def get_context(n, pl):
    node = pl[-1]
    i = 0
    contlist = []
    while i < n and len(node.prev)>0:
        node = get_node(node, 0)
        contlist.append(node)
        i+=1
    assert len(node.prev)==0
    return contlist
            

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
    c = get_context(300, [node])
    toks = [con.token_idx for con in c]
    toks.reverse()
    toks.append(node.token_idx)
    s = tokenizer.batch_decode([toks])[0]
    s = s.replace("</s>", "")
    s = s.replace("de_DE", "")
    s = s.replace("en_XX", "")
    return s.strip()

def get_plist_str(plist):
    toks = [con.token_idx for con in plist]
    toks.reverse()
    s = tokenizer.batch_decode([toks])[0]
    s = s.replace("</s>", "")
    s = s.replace("de_DE", "")
    s = s.replace("en_XX", "")
    return s.strip()

def get_plist_sco(plist):
    tot = 0
    for p in plist:
        tot+=p.score
    return tot/len(plist)

def find_paths(root, graph):
    global nodeset
    #print(root.token_str)
    if len(root.prev) == 0:
        yield [root]

    seen = []
    for c in range(0, len(root.prev)):
        child = get_node(root, c)
        if child.uid in seen:
            continue
        nodeset.add(child.uid)
        #if len(seen)>1:
            #print("maybe not bug")
        seen.append(child.uid)
        for path in find_paths(child, graph):
            yield [root] + path

def get_uid_str(plist):
    res = ""
    for p in plist:
        res+=str(p.uid)
    return res

def plist_equal(l1, l2):
    return get_uid_str(l1)==get_uid_str(l2)

def remove_dups(allplists):
    if len(allplists) ==0:
        return []
    uidlist = []
    res = []
    for plist in allplists:
        uidlist.append(get_uid_str(plist))
    i = 0
    while(i<len(allplists)-1):
        if uidlist[i] in uidlist[i+1:]:
            i+=1
            continue
        else:
            res.append(allplists[i])
        i+=1
    res.append(allplists[-1])
    return res
    
STOP = 100    
nodeset = set()
tot = 0
# TODO, wait, does this just mean that recombination doesn't do anything
# use candidates from this as a theoretical bound on what lattices can do
def get_all_possible_candidates(fname, needbase=True):
    global nodeset
    global tot 
    if needbase:
        graph = get_graph(BASE+fname)
    else:
        graph = get_graph(fname)
    # if we only want graphs with less than glim nodes
    scores =  []
    cands = []
    ref = graph.reference
    src = graph.document
    fullplist = []
    for e in graph.ends:
        generated = 0
        for p in find_paths(e, graph):
            if generated == STOP:
                break
            fullplist.append(p)
            generated+=1
 
    nodeset = set()
    fullplist = remove_dups(fullplist)

    if tot%300==0:
        print(tot)
    for plist in fullplist:
        scores.append(get_plist_sco(plist))
        cands.append(get_plist_str(plist))
    tot+=1
    
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
        if explode=="True":
            try:
                s, c, r, d = get_all_possible_candidates(fnam)
            except:
                s, c, r, d = [], [], "", ""
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

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str)
    
    parser.add_argument('-device', type=str, default='cuda:2')
    parser.add_argument('-exploded', type=str, default="False")
    parser.add_argument('-path_output', type=str, default="mtn1_fr-en_bfs_recom_2_-1_False_0.4_True_False_4_5_zip_-1_0.0_0.9")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = process_args()
    BASE = "./custom_output/data/"+args.path_output+"/"
    if BASE =="":
        BASE = "./custom_output/data"+"/mtn1_fr-en_bfs_recom_2_-1_False_0.4_True_False_4_5_zip_-1_0.0_0.9/"

    if args.dataset=='en_de':
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

    #run_save_graphs()
    latticecandjson = process_save_all_graphs(args.exploded)
    print(latticecandjson[0])
    sname = args.path_output
    import json
    name = "./candoutputs/"+sname+".jsonl"
    if args.exploded=="True":
        name = "./candoutputs/exploded"+sname+".jsonl"
    tmpfile = open(name, "w")
    for l in latticecandjson:
        tmpfile.write(json.dumps(l))
        tmpfile.write('\n')
    tmpfile.close()