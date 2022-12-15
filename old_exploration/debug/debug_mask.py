from latmask_bert_models import LatticeBertModel
from transformers import AutoConfig
import torch
import pickle

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class LinearPOSBertV1(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = LatticeBertModel(AutoConfig.from_pretrained('bert-base-cased'))
        self.probe = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()
  
    def forward(self, sentences, pos_ids=None, attmasks=None):
        with torch.no_grad(): # no training of BERT parameters
            if pos_ids==None:
                word_rep, sentence_rep = self.bert(sentences, return_dict=False)
            else:
                print('YO')
                word_rep, sentence_rep = self.bert(sentences, position_ids=pos_ids, encoder_attention_mask=attmasks, attention_mask=attmasks, return_dict=False)
        return self.probe(word_rep)

def check_accuracy(setpred, setlabels):
    cor = 0
    tot = 0
    for i in range(0, len(setpred)):
        ex = setpred[i]
        for j in range(0, len(ex)):
            if sum(setlabels[i][j])==0:
                continue
            tot+=1
            if torch.argmax(ex[j])==torch.argmax(setlabels[i][j]):
                cor+=1
    return cor/tot

# fix pos_ids
def fix_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            if i>0 and p[i]==0:
                p[i] = i
    return cop

def mod_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            if i>0:
                p[i] = i
    return cop
    
# make sure that there's attention on the diagonals as well
def fix_att_masks (attmasks):
    for a in attmasks:
        for i in range(0, len(a)):
            #assert a[i][i] == 0
            a[i][i] = 1
    return attmasks


if __name__ == "__main__":
    print("Starting up")
    sents = torch.load("./torchsaved/sents.pt").to(device)
    posids = torch.load("./torchsaved/posids.pt").to(device)
    latattmasks = torch.load("./torchsaved/fixlatattmasks.pt").to(device)
    onemask = torch.ones_like(latattmasks)
    ylabels = torch.load("./torchsaved/betterposlatposylabels.pt").to(device)
    latattmasks = fix_att_masks(latattmasks)
    print("Loading Model")
    posbmodel = LinearPOSBertV1(44)
    t = torch.load("./a3-distrib/ckpt/posbert.pth").to(device)
    posbmodel.load_state_dict(t)
    posbmodel.eval()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated("cuda:1"))

    print("Starting Evaluation")
    # hard default posids used
    pred = posbmodel(sents, fix_posids(posids), onemask)

    print("Accuracy")
    print(check_accuracy(pred, ylabels))





