import torch
import torch.nn as nn
from transformers import AutoModel
# lattice encoding and normal candidate encoding
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class LinearLatticeBert(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-cased')
        self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()
  
    def forward(self, sentences, posids):
    
        with torch.no_grad(): # no training of BERT parameters
            bertout = self.bert(sentences, position_ids=posids, return_dict=True, output_hidden_states=True)
        return bertout
    
class LinearPOSBertV1(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        #self.bert = LatticeBertModel(AutoConfig.from_pretrained('bert-base-cased'))
        self.bert = AutoModel.from_pretrained('bert-base-cased')
        self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()
  
    def forward(self, sentences, pos_ids=None, attmasks=None):
        with torch.no_grad(): # no training of BERT parameters
            word_rep, sentence_rep = self.bert(sentences, position_ids=pos_ids, encoder_attention_mask=attmasks, attention_mask=attmasks, return_dict=False)
        return self.probe(word_rep)