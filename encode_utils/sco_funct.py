import math

def default_scofunct (node, unused, norm):
    global pcnt
    try:
        return node.score
    except:
        pcnt+=1
        return 0

#npcnt = 0
def addprob (node, unused, norm):
    global pcnt, npcnt
    if "prob" in node.keys():
        #pcnt+=1
        return math.log(node['prob']) + node['score']
    else:
        #npcnt+=1
        return 1

WEIGHT= .67
def weightaddprob (node, unused, norm):
    global pcnt, npcnt
    if hasattr(node, "prob"):
        return math.log(node.prob) + WEIGHT*node.score
    else:
        return 1
    
DIVWEIGHT = 1
# should this be more complex
def weightadddiverse (node, used, norm):
    global usedlist
    if hasattr(node, "prob"):
        #pcnt+=1
        if node.token_idx in used:
            # TODO maybe we need to add an element that factors in position as well
            return math.log(node.prob) + WEIGHT*node.score - DIVWEIGHT
        return math.log(node.prob) + WEIGHT*node.score
    else:
        #npcnt+=1
        return 1

def multprob (node):
    if "prob" in node.keys():
        #pcnt+=1
        return node['prob'] * node['score']
    else:
        #npcnt+=1
        return 1