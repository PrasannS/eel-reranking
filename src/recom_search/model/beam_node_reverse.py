from typing import List

class ReverseNode():
    def __init__(self, oldnode):
        self.uid = oldnode.uid
        self.prob = oldnode.prob
        self.token_idx = oldnode.token_idx
        self.token_str = oldnode.token_str
        self.nextlist = []
        self.next_scores = []
        self.next_ids = []
        self.pos = -1

