import torch

# get adjacency from flat canvas of tokens, with previous tokens
def adj_mat(flat_canv, mlen):
    mask = torch.zeros((mlen, mlen))
    # have a cutoff after the calculated limit to circumvent extra computation
    # and potential bugs
    for row in range(mlen):
        for p in flat_canv[row].prevs:
            # TODO verify correctness of this
            if p.canvpos<mlen:
                mask[row][p.canvpos] = 1
    return mask

# ensure that token canvas has the right values for mask creation
def mask_prep_canv(canv):
    maxpos = -1
    # ensure that prevs are all valid
    for i in range(len(canv)):
        node = canv[i]
        node.prevs = []
    for i in range(len(canv)):
        for n in canv[i].nextlist:
            assert n in canv
            n.prevs.append(canv[i])
    # we're taking in DLReverseNodes
    ind = 0
    for c in canv:
        # maybe should be an assert instead to ward off weirdness?
        c.canvpos = ind
        maxpos = max(maxpos, c.pos)
        # TODO only include as a test, remove later
        for prev in c.prevs:
            assert prev in canv
        ind+=1
    return maxpos  

MAXTOKS = 512
# get connectivity matrix from flat canvas of tokens
# by exponentiating reverse connected adjacency
def get_connect_mask(flat_canv, posadd):
    # TODO something to get length, while considering space from inp tokens
    mlen = min(MAXTOKS-(posadd+1), len(flat_canv))
    maxpos = mask_prep_canv(flat_canv)
    back_adjac = adj_mat(flat_canv, mlen)
    tot = back_adjac
    tmp = back_adjac
    # keep on going until all nodes hit the back
    while torch.sum((tot[:, 0]>0))<(mlen-1):
        tmp = torch.mm(back_adjac, tmp)
        tot += tmp
    tot = tot+ torch.eye(mlen)
    return (tot>0).int()

def get_causal_mask(flatcanv, posadd):
    custmax = get_connect_mask(flatcanv, posadd)
    # everything starts off as 1s
    res = torch.ones((MAXTOKS, MAXTOKS))
    # causual on the right side
    res[:, posadd:] = 0
    addpos = len(custmax)+posadd
    res[posadd:addpos, posadd:addpos] = custmax
    return res

    