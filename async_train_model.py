from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import csv
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
import sys
import random
import pandas as pd
from rerank_score_cands_new import load_cands
import numpy as np
from comet import download_model, load_from_checkpoint
import pickle
from sklearn.utils import shuffle
#from distill_comet import XLMCometRegressor

csv.field_size_limit(sys.maxsize)
xlm_tok = AutoTokenizer.from_pretrained('xlm-roberta-base')

print("starting")

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

def load_cometqe_data():
    with open('processeddata/germanlatmasks.pkl', 'rb') as f:
        masks = pickle.load(f)

    with open('processeddata/germanlatinps.pkl', 'rb') as f:
        xinps = pickle.load(f)

    with open('processeddata/germanlatlabels.pkl', 'rb') as f:
        yinps = pickle.load(f)
    return masks, xinps, yinps

mdata, xdata, ydata = load_cometqe_data()
print("data loaded")

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, sentences, labels, masks):
        assert len(sentences) == len(labels)
        self.sentences = sentences
        self.labels = labels
        self.masks = masks

    def __getitem__(self, i):
        return self.sentences[i], self.labels[i], self.masks[i]

    def __len__(self):
        return len(self.sentences)

def collate_custom(datafull):
    #print(len(datafull[0]))
    data = [torch.tensor(d[0]) for d in datafull]
    masdata=  [d[2] for d in datafull]
    labels = [d[1] for d in datafull]
    max_len = max([x.squeeze().numel() for x in data])
    data = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in data]
    data = torch.stack(data).to(device)
    # TODO just a normal mask for now
    #masdata = [torch.ones_like(m) for m in masdata]
    masdata = [torch.nn.functional.pad(x, pad=(0, max_len - x[0].numel(), 0, max_len - x[0].numel()), mode='constant', value=0) for x in masdata]
    masdata = torch.stack(masdata).to(device)
    return data, torch.tensor(labels).to(device), masdata

subset = 32*100
xdata = xdata[:subset]
ydata = ydata[:subset]
mdata = mdata[:subset]

trainloader = DataLoader(RegressionDataset(xdata, ydata, mdata), batch_size=32, shuffle=False, collate_fn=collate_custom)

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm_
def train(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, device, clip_value=2):
    print("Total steps :", epochs*len(train_dataloader))
    best_loss = 1e10
    
    for epoch in range(epochs):
        if epoch%1==0:
            print("EPOCH ", epoch)
            print("-----")
            print(best_loss)
        model.train()
        cbest = 1e10
        lostot = 0
        loscnt = 0
        for step, batch in enumerate(train_dataloader): 
            batch_inputs, batch_labels, batch_masks = \
                               tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)
            loss = loss_function(outputs.squeeze(), 
                             batch_labels.squeeze())
            lostot+=loss
            loscnt+=1
            if step%5==0:
                #print(loss)  
                if loscnt>0:
                    print(lostot/loscnt)
                    cbest = min(float(lostot/loscnt), cbest)
                    best_loss = min(best_loss, cbest)
                    #print("cbest, ", cbest)
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
        cbest = min(float(lostot/loscnt), cbest)
        best_loss = min(best_loss, cbest)
        print("cbest, ", cbest)
        if epoch%5==0:
            #torch.save(model.state_dict(), "torchsaved/germantest"+str(epoch)+".pt")
            ""
    return model

model = XLMCometRegressor(drop_rate=0.1)
#model.load_state_dict(torch.load("./torchsaved/germanlat9.pt"))

print("model loaded")

# mask that can be used for rank losses to happen in O(N)
vmask = (torch.triu(torch.ones(32, 32))*2-torch.ones(32, 32))*-1
vmask = vmask.to(device)
mse = nn.MSELoss()

# margin based rank loss, TODO sanity check again to make sure it's fine
def rank_loss(preds, golds):
    totloss = 0
    lim = int(len(preds))
    for i in range(1, lim):
        # for margin
        margin = (golds - torch.roll(golds, i))*vmask[i]
        diff = ((preds - torch.roll(preds, i))-margin)*vmask[i]
        diff[diff<0] = 0
        totloss+=torch.sum(diff)
    return totloss + mse(preds, golds)

def run_model_train_params(learn_r, epochs, loader, mod, loss):
    optimizer = AdamW(mod.parameters(),
                      lr=learn_r,
                      eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                     num_warmup_steps=0, num_training_steps=epochs*len(loader))
    model = train(mod, optimizer, scheduler, loss, epochs, 
                  loader, device, clip_value=2)

print("start training")
print(ydata)
# converge on this first
#run_model_train_params(1e-5, 2, trainloader, model, mse)
#run_model_train_params(1e-5, 10, trainloader, model, rank_easy)
run_model_train_params(1e-5, 200, trainloader, model, mse)
#run_model_train_params(1e-5, 100, trainloader, model, rank_easy)

#run_model_train_params(5e-5, 5, trainloader, model, rank_easy)
#run_model_train_params(1e-5, 10, trainloader, model, rank_easy)
#run_model_train_params(1e-5, 20, trainloader, model, rank_loss)

#run_model_train_params(1e-5, 40, trainloader, model, rank_loss)
# nohup python -u async_train_model.py &