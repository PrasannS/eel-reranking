from src.recom_search.model.beam_node_reverse import ReverseNode
from generate_utils.lattice_cands import get_node, get_graph, get_all_possible_candidates
import os
import pickle
import pandas as pd

def reverse_graph(graph):
    allnodes = {}
    for g in graph.ends:
        update_graph(allnodes, g)
    allnodes['input'] = graph.document
    allnodes['ref'] = graph.reference
    return allnodes

def update_graph(nodedict, node):
    # base case, reached the start
    if len(node.prev)==0:
        if 'root' not in nodedict:
            nodedict['rootid'] = node.uid
            nodedict['root'] = nodedict[node.uid]
        else:
            # if there are multiple roots add logic for that
            assert nodedict['rootid'] == node.uid
        return
    # for ends
    if node.uid not in nodedict:
        nodedict[node.uid] = ReverseNode(node)
    for n in range(0, len(node.prev)):
        #pid = node.prev[n]
        pnode = get_node(node, n)
        first = False
        # check if in node dictionary, first time visiting
        if pnode.uid not in nodedict:
            # add new node
            # TODO offload long constructor
            nodedict[pnode.uid] = ReverseNode(pnode)
            first = True
        prevnode = nodedict[pnode.uid]
        if node.uid not in prevnode.next_ids:
            prevnode.nextlist.append(nodedict[node.uid])
            prevnode.next_scores.append(node.score)
            prevnode.next_ids.append(node.uid)
        else:
            # this means that there's a cycle
            return
        # pass on recursion if this node hasn't been seen before
        if first:
            update_graph(nodedict, pnode)

def num_choices (ndict):
    cnt = 0
    for n in ndict.keys():
        if 'root' in n or 'input' in n or 'ref' in n:
            continue
        curnode = ndict[n]
        if len(curnode.nextlist)>1:
            cnt+=1
    return cnt

def reverse_save_graphs(path_output, foldername):
    BASE = "./custom_output/data/"+path_output+"/"
    TGT = "./outputs/graph_pickles/"+foldername+"/"
   
    # get location of all pickle files
    paths = os.listdir(BASE)
    if os.path.exists(TGT)==False:
        os.mkdir(TGT)
    ind = 0
    for pat in paths:
        curgraph = get_graph(BASE+pat)
        restmp = reverse_graph(curgraph)

        assert len(restmp.keys())>0

        filehandler = open(TGT+str(ind), 'wb') 
        pickle.dump(restmp, filehandler)
        print("Num places where path diverges")
        print(num_choices(restmp))
        print("Num expanded nodes")
        print(len(restmp.keys())-2)
        #print(TGT+pat)
        ind+=1

def reverse_df_graphs(inpnames):
    data = pd.read_csv(inpnames+".csv")
    if os.path.exists(inpnames+"reversed")==False:
        os.mkdir(inpnames+"reversed")
    ind = 0
    for d in data['fname']:
        curgraph = get_graph(d)
        restmp = reverse_graph(curgraph)
        assert len(restmp.keys())>0
        filehandler = open(inpnames+"reversed/"+str(ind), 'wb') 
        pickle.dump(restmp, filehandler)
        ind+=1

def explode_df_graphs(inpnames):
    data = pd.read_csv(inpnames+".csv")
    if os.path.exists(inpnames+"exploded")==False:
        os.mkdir(inpnames+"exploded")
    ind = 0
    for d in data['fname']:
        acands = get_all_possible_candidates(d, False)
        filehandler = open(inpnames+"exploded/"+str(ind), 'wb') 
        pickle.dump(acands, filehandler)
        print(ind)
        ind+=1

def explode_graphs(reversedir, newname):
    srcbase = "./custom_output/data/"+reversedir+"/"
    newbase = "./outputs/graph_pickles/"+newname+"/"
    graphpaths = os.listdir(srcbase)

    if os.path.exists(newbase)==False:
        os.mkdir(newbase)
    ind = 0
    for g in graphpaths:
        try:
            acands = get_all_possible_candidates(srcbase+g, False)
            filehandler = open(newbase+str(ind), 'wb') 
            pickle.dump(acands, filehandler)
        except:
            print("Recursion Failed")
        print(ind)
        ind+=1

if __name__ == "__main__":
    # TODO set up logic to retrieve graph given filename
    # reverse_save_graphs("mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9")
    #reverse_save_graphs("mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.902_0.0_0.9")
    #reverse_df_graphs("german_fnames")
    #reverse_df_graphs("french_fnames")
    # russian
    # reverse_save_graphs("sum_xsum_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.903_0.0_0.9", "nounsum_reversed")
    explode_graphs("sum_xsum_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.903_0.0_0.9", "nounsum_exploded")
    #print("starting french")
    #explode_df_graphs("french_fnames")
    #print("starting german")
    #explode_df_graphs("german_fnames")
    