from transformers import AutoTokenizer
from src.recom_search.model.beam_node_reverse import ReverseNode


bert_tok = AutoTokenizer.from_pretrained("bert-base-cased")


def node_from_idx(t1, ind, idval = None):
    if idval:
        return ReverseNode(None, {'uid':str(t1)+str(ind), 'token_idx':t1, 'token_str':bert_tok.decode(t1), 'prob':0})
    else:
        return ReverseNode(None, {'uid':str(t1)+str(ind)+" "+str(idval), 'token_idx':t1, 'token_str':bert_tok.decode(t1), 'prob':0})
       
def complete_path(gdict, toks, i, prev, pind):
    for j in range(i, len(toks)):
        cur1 = node_from_idx(toks[j], j, pind)
        gdict[cur1.uid] = cur1
        prev.nextlist.append(cur1)
        prev.next_scores.append(cur1.prob)
        prev.next_ids.append(cur1.uid)
        prev = cur1
        
def tlist_check(tlists, i):
    # flourless chocolate cake
    check = tlists[0][i]
    for t in tlists:
        if t[i] == check:
            continue
        else:
            return False
    return True

def create_toy_graph(slist, tok):
    tokslist = [tok(s).input_ids for s in slist]
    
    prev = None
    merged = False
    gdict = {}
    i = 0
    # while they're the same just decode the same
    while tlist_check(tokslist, i):
        # make node at current position, if root then set as root
        cur1 = node_from_idx(tokslist[0][i], i)
        gdict[cur1.uid] = cur1
        if i==0:
            gdict['root'] = cur1
        else:
            prev.nextlist.append(cur1)
            prev.next_scores.append(cur1.prob)
            prev.next_ids.append(cur1.uid)
        prev = cur1
        i+=1
    
    puid = prev.uid
    # we've now split, finish off the last 2 sides one at a time
    for j in range(len(tokslist)):
        complete_path(gdict, tokslist[j], i, gdict[puid], j)
        
    return gdict