import torch
# TODO go into github to get the older version of this file

# TODO more sanity checking on whether or not we're recording the right number of branches     
def get_adjac_mat(pgraph):
    res = []
    for node in pgraph:
        tmp = []
        for p in pgraph:
            if str(p['id']) in node['nexts']:
                tmp.append(1)
            else:
                tmp.append(0)
        res.append(torch.tensor(tmp))
    return torch.stack(res)

def get_poslist(processed):
    res = []
    for p in processed:
        res.append(p['pos'])
    return res

def conmat(adj, longestpath):
    res = torch.zeros_like(adj)
    adj = torch.triu(adj)
    start = adj
    for i in range(0, longestpath):
        if i>0:
            start = torch.mm(start, adj)
        res = res + start
    return res

# fixes input to now account for [SEP] and [CLS] I believe
def correct_mask_sep(mat):
    for i in range(len(mat)):
        mat[i][i] = 1
    # TODO need to remove for stuff
    # fixed = torch.nn.functional.pad(input=mat, pad=(1, 1, 1, 1), mode='constant', value=1)
    return mat

def connect_mat(pgraph):
    conm = conmat(get_adjac_mat(pgraph), max(get_poslist(pgraph)))
    conm = conm + torch.transpose(conm, 0, 1)

    cmat =  (conm>0).float()
    # handle things to be able to take when really big input
    respad = None
    MAX =512
    # we're within 498
    if len(cmat) < MAX-1:
        # get the version with CLS and SEP
        cmat = correct_mask_sep(cmat)
        # copy that onto fixed input: the rest is padding
        respad = torch.zeros((MAX, MAX))
        cpmax = len(cmat)
        for i in range(cpmax):
            for j in range(cpmax):
                respad[i][j] = cmat[i][j]
    else:
        # TODO made changes here
        # we're over the limit, first get truncated version
        respad = torch.zeros((MAX, MAX))
        # make truncated matrix
        for i in range(MAX):
            for j in range(MAX):
                respad[i][j] = cmat[i][j]
        # do CLS / SEP tokens on that 
        respad = correct_mask_sep(respad)
        
    #respad[respad==0] = -float('inf')
    return respad #ones_padding(respad)