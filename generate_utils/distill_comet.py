from genericpath import exists
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import csv
import sys
import random
import pandas as pd
from .rerank_score_cands_new import load_cands
import numpy as np
from tfr_models import download_model, load_from_checkpoint
import pickle
from sklearn.utils import shuffle

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
csv.field_size_limit(sys.maxsize)
xlm_tok = AutoTokenizer.from_pretrained('xlm-roberta-base')
loss_function = nn.MSELoss()

# Encoder-Decoder Model Embedding, Add a Weighted Layer at the end that leads to regression
class XLMCometRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.1):
        # TODO should we be freezing layers?
        super().__init__()
        
        self.xlmroberta = AutoModel.from_pretrained('xlm-roberta-base')
        # Num labels 1 should just indicate regression (?)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.xlmroberta.config.hidden_size, 1), 
        )
        self.to(device)
        
    def forward(self, input_ids, attention_masks):
        # don't finetune xlmroberta model
        #with torch.no_grad():
        word_rep, sentence_rep = self.xlmroberta(input_ids, attention_mask=attention_masks, encoder_attention_mask=attention_masks, return_dict=False)
        # use the first <s> token as a CLS token, TODO experiment with using the sum of 
        # ensure padding not factored in
        word_rep = word_rep*(input_ids>0).unsqueeze(-1)
        outputs = self.regressor(torch.sum(word_rep, 1))
        #print("Shape: ", outputs.shape)
        return outputs
    
# Returns Token Scores For Each (Sum becomes Regression)
class XLMCometEmbeds(nn.Module):
    
    def __init__(self, drop_rate=0.1):
        # TODO should we be freezing layers?
        super().__init__()
        
        self.xlmroberta = AutoModel.from_pretrained('xlm-roberta-base')
        # Num labels 1 should just indicate regression (?)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.xlmroberta.config.hidden_size, 1), 
        )
        self.to(device)
        
    def forward(self, input_ids, positions, attention_masks):
        # don't finetune xlmroberta model
        #with torch.no_grad():
        word_rep, sentence_rep = self.xlmroberta(input_ids, position_ids = positions, attention_mask=attention_masks, encoder_attention_mask=attention_masks, return_dict=False)
        # use the first <s> token as a CLS token, TODO experiment with using the sum of 
        # ensure padding not factored in
        word_rep = word_rep*(input_ids>0).unsqueeze(-1)
        outputs = self.regressor(word_rep)
        #print("Shape: ", outputs.shape)
        return outputs

# construct inputs from dataframe
def get_test_inputs(inps, hyps, savedat=False):
    assert len(inps)==len(hyps)
    inpdf = pd.DataFrame()
    inpdf['inp'] = inps
    inpdf['hyp'] = hyps
    xinp = []
    maskinp = []
    if exists("torchsaved/latxinp.pkl") and savedat:
        with open("torchsaved/latxinp.pkl", "rb") as f:
            xinp = pickle.load(f)
        with open("torchsaved/latmaskinp.pkl", "rb") as f:
            maskinp = pickle.load(f)

        return xinp, maskinp
    
    print("total = ", len(inpdf))
    for index, row in inpdf.iterrows():
        if index%1000==0:
            print(index)
        #print(row['c1'], row['c2'])
        # will need to make a custom mask (maybe) so that inputs from both languages are encoded separately
        toktmp = xlm_tok(row['inp']).input_ids
        lent = len(toktmp)
        hyptmp = xlm_tok(row['hyp']).input_ids
        toktmp.extend(hyptmp)
        mask = torch.ones(len(toktmp), len(toktmp))
        # make causal mask
        mask[:lent, lent:] = 0
        mask[lent:, lent:] = torch.tril(mask[lent:, lent:])
        xinp.append(toktmp)
        maskinp.append(mask)
    if savedat:
        with open("torchsaved/latxinp.pkl", "wb") as f:
            pickle.dump(xinp,f )
        with open("torchsaved/latmaskinp.pkl", "wb") as f:
            pickle.dump(maskinp,f )
 
    return xinp, maskinp

class RegressionDataset(Dataset):
    def __init__(self, sentences, masks):
        assert len(sentences) == len(masks)
        self.sentences = sentences
        self.masks = masks

    def __getitem__(self, i):
        return self.sentences[i], self.masks[i]

    def __len__(self):
        return len(self.sentences)

def collate_custom(datafull):
    #print(len(datafull[0]))
    data = [torch.tensor(d[0]) for d in datafull]
    masdata=  [d[1] for d in datafull]
    max_len = max([x.squeeze().numel() for x in data])
    data = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in data]
    if max_len>508:
        data = [d[:508] for d in data]
    data = torch.stack(data).to(device)
    # just a normal mask for now
    masdata = [torch.ones_like(m) for m in masdata]
    masdata = [torch.nn.functional.pad(x, pad=(0, max_len - x[0].numel(), 0, max_len - x[0].numel()), mode='constant', value=0) for x in masdata]
    if max_len>508:
        masdata = [d[:508, :508] for d in masdata]
    masdata = torch.stack(masdata).to(device)
    
    return data, masdata

def load_distill_model(mname="./torchsaved/maskedcont4.pt"):
    model = XLMCometRegressor(drop_rate=0.1)
    model.load_state_dict(torch.load(mname))
    model.eval()
    return model

def evaluate(model, tdataloader, device):
    model.eval()
    preds = []
    ind = 0
    for batch in tdataloader:
        if ind%50==0:
            print(ind)
        batch_inputs, batch_masks = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        preds.extend(list(outputs.squeeze()))
        ind+=1
    with open("./torchsaved/lattestsaved.pkl", "wb") as f:
        pickle.dump(preds, f)
    return preds

def run_distill_comet(inps, hyps, model):
    xtest, mtest = get_test_inputs(inps, hyps)
    testloader = DataLoader(RegressionDataset(xtest, mtest), batch_size=32, shuffle=False, collate_fn=collate_custom)
    results =  evaluate(model, testloader, device)
    return [float(r) for r in results]
