from typing import List

class ReverseNode():
    def __init__(self, oldnode=None, infodict=None):
        if oldnode:
            self.uid = oldnode.uid
            self.prob = oldnode.prob
            self.token_idx = oldnode.token_idx
            self.token_str = oldnode.token_str
        else:
            self.uid = infodict['uid']
            self.prob = infodict['prob']
            self.token_idx = infodict['token_idx']
            self.token_str = infodict['token_str']
        self.nextlist = []
        self.next_scores = []
        self.next_ids = []
        self.pos = -1

