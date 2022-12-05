from heapq import heappop, heappush
from collections import UserDict, defaultdict
from heapq import heappop
from turtle import color
from pyvis.network import Network
import networkx as nx
import tqdm
import pickle
import os

def get_node (node, ind):
    return node.hash.retrieve_node(node.prev[ind])

def construct(frontier, completed, tokenizer):
    if frontier and isinstance(frontier[0], tuple):
        frontier = [x[1] for x in frontier]
    if isinstance(completed[0], tuple):
        completed = [x[1] for x in completed]
    edges = {}
    nodes = {}

    # record all of the children nodes, top_down['root'] = [root_node_uid]
    top_down = defaultdict(list)
    seen = {}

    def dfs(node):
        if not node:
            return
        node_uid = node.hash.find_root_node_uid(node.uid)
        if node_uid in seen:
            return
        seen[node_uid] = True
        node = node.hash.retrieve_node(node_uid)
        nodes[node.uid] = {
            'uid': node_uid,
            'step': node.length,
            'token_idx': node.token_idx,
            'tok':tokenizer.decode(node.token_idx),
            'prob': node.prob,
            'acc_score': node.score,
            'length': node.length,
            'compl': True
        }
        if not node.prev:
            top_down['root'] = [node_uid]
            return
        my_prev, my_prev_score = node.prev, node.prev_score
        for p, ps in zip(my_prev, my_prev_score):
            p_uid = node.hash.find_root_node_uid(p)
            # p_node = self.hash.retrieve_node(p_uid)
            edge_info = {
                'src': p_uid,
                'tgt': node_uid,
                'score': ps,
                'prob':ps
            }
            edges[f"{p_uid}_{node_uid}"] = edge_info
            top_down[p_uid].append(node_uid)
            # edges.append(edge_info)

        


        prevs = node.prev
        _, prevs = node.hash.retrieve_group_nodes(prevs)
        for p in prevs:
            dfs(p)
    
    for comp in completed:
        dfs(comp)
    print("ENDS")
    print(len(completed))
    if frontier:
        for uncomp in frontier:
            dfs(uncomp)
    return edges, nodes, top_down

from collections import Counter

def count_children_number(top_down):
    counter = Counter()
    start = top_down['root'][0]

    def dfs(node_id):
        if node_id in counter:
            return counter[node_id]
        kids = top_down[node_id]
        counter[node_id] = 1
        for k in kids:
            counter[node_id] += dfs(k)
        return counter[node_id]
    total_cnt = dfs(start)
    return counter, total_cnt

def assign_pos(node_uid, children_cnt, start_pos, edges, nodes, top_down,location_map):

    location_map[node_uid] = [nodes[node_uid]['length'], start_pos]
    kids = top_down[node_uid]
    offset = 0
    for k in kids:

        assign_pos(k, children_cnt, start_pos + offset , edges, nodes, top_down,location_map)
        offset += children_cnt[k]
    

def visualize_fixed(output_dict, tokenizer):
    
    net = Network(height='100%', width='100%')
    # first draw the incomplete trees, and then completed so we can overide the completed states
    # completed = output_dict['completed']
    completed = output_dict.ends
    # frontier = output_dict['frontier']
    edges, nodes, top_down = construct(None, completed, tokenizer)
    children_cnt, total_cnt = count_children_number(top_down)
    print('...')
    # start span, end span
    # assign x and y to all nodes, x is time step, y is the position
    location_map = {}
    assign_pos(top_down['root'][0], children_cnt,0,edges, nodes, top_down,location_map)
    
    # draw nodes
    for node in nodes.values():
        if node['compl']:
            c = 'red'
        else:
            c = 'blue'
        net.add_node(node['uid'], label=f"{node['tok']}", shape='dot', x=location_map[node['uid']][0]*150, y=location_map[node['uid']][1] * 10, color=c, size = 2.5)


    # draw edges
    for edge in edges.values():
        form = "{:.1f}".format(edge['prob'])
        net.add_edge(edge['src'], edge['tgt'], title=form, width=2 * edge['prob'], arrowStrikethrough=False)  #arrowStrikethrough=False

    for n in net.nodes:
        n.update({'physics': False})
    return net

if __name__ == "__main__":
    # # execute only if run as a script
    # prefix = 'sum'
    # suffix = '.pkl'
    # suffix = 'astar_15_35_False_0.4_True_False_4_5_imp_False_0.0_0.9.pkl'
    # files = os.listdir('vizs')
    # files = [f for f in files if f.endswith(suffix) and f.startswith(prefix)]
    print(os.getcwd())
    with open('custom_output/data/mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9/mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9_0_L\'affaire-.pkl','rb') as fd:
        output_dict = pickle.load(fd)
    print(output_dict)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-one-to-many-mmt')

    files = os.listdir('custom_output/data/mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9')
    files = [x for x in files if x.endswith('pkl')]
    print("NUMFILES")
    print(len(files))
    for f in files:
        name = ".".join(f.split('.')[:-1])
        with open('custom_output/data/mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9/'+f,'rb') as fd:
            output_dict = pickle.load(fd)
        try:
            net = visualize_fixed(output_dict, tokenizer)
            net.show(f"vizs/new-{name}.html")
            print(name)
        except:
            print("Recursion error")
        #break

