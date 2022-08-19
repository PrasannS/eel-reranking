from src.recom_search.model.beam_node_reverse import ReverseNode
from lattice_cands import get_node, get_graph
import os
import pickle

def reverse_graph(graph):
    allnodes = {}
    for g in graph.ends:
        update_graph(allnodes, g)
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
        pid = node.prev[n]
        pnode = get_node(node, n)
        first = False
        # check if in node dictionary, first time visiting
        if pid not in nodedict:
            # add new node
            # TODO offload long constructor
            nodedict[pid] = ReverseNode(pnode)
            first = True
        prevnode = nodedict[pid]
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
        if 'root' in n:
            continue
        curnode = ndict[n]
        if len(curnode.nextlist)>1:
            cnt+=1
    return cnt

def reverse_save_graphs(path_output):
    BASE = "./custom_output/data/"+path_output+"/"
    TGT = "./reverse_graphs/"+path_output+"/"
   
    # get location of all pickle files
    paths = os.listdir(BASE)
    if os.path.exists(TGT)==False:
        os.mkdir(TGT)

    for pat in paths:
        curgraph = get_graph(BASE+pat)
        restmp = reverse_graph(curgraph)

        assert len(restmp.keys())>0

        filehandler = open(TGT+pat, 'wb') 
        pickle.dump(restmp, filehandler)
        print("Num places where path diverges")
        print(num_choices(restmp))
        print("Num expanded nodes")
        print(len(restmp.keys())-2)
        print(TGT+pat)

if __name__ == "__main__":
    # TODO set up logic to retrieve graph given filename
    # reverse_save_graphs("mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9")
    reverse_save_graphs("mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.902_0.0_0.9")