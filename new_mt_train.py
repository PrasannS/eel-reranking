# Pipeline for training to get a causal model that matches some metric
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import csv
import sys
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm_
from process_traindata import create_sortedbatch_data
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

csv.field_size_limit(sys.maxsize)

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
        word_rep, sentence_rep = self.xlmroberta(input_ids, attention_mask=attention_masks, encoder_attention_mask=attention_masks, return_dict=False)
        # ensure padding not factored in
        word_rep = word_rep*(input_ids>0).unsqueeze(-1)
        # NEW average embeddings based on input length in case length variance was an issue
        avg_embed = torch.sum(word_rep, 1)/torch.sum(input_ids>0)

        outputs = self.regressor(avg_embed)
        return outputs

class RegressionDataset(Dataset):
    def __init__(self, sentences, labels, hypstarts):
        assert len(sentences) == len(labels)
        self.sentences = sentences
        self.labels = labels
        self.hypstarts = hypstarts

    def __getitem__(self, i):
        return self.sentences[i], self.labels[i], self.hypstarts[i]

    def __len__(self):
        return len(self.sentences)

# process data for a single batch
def collate_custom(datafull):
    # Extract out data from input tuples
    with torch.no_grad():
        data = [torch.tensor(d[0]) for d in datafull]
        hypstarts =  [d[2] for d in datafull]
        labels = [d[1] for d in datafull]
        # Figure out how much to pad by, then pad up inp tokens
        max_len = max([x.squeeze().numel() for x in data])
        data = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in data]
        data = torch.stack(data).to(device)
        
        # Keep causal mask or 
        if CAUSAL:
            masdata = []
            for mind in range(len(data)):
                # TODO need to sanity check mask once
                totoks = data[mind].squeeze().numel() 
                msktmp = torch.ones((totoks, totoks))
                hind = hypstarts[mind]
                msktmp[:hind, hind:] = 0
                msktmp[hind:, hind:] = torch.tril(msktmp[hind:, hind:])
                masdata.append(msktmp)

            # TODO move custom causal logic here
            masdata = [torch.nn.functional.pad(x, pad=(0, max_len - x[0].numel(), 0, \
                        max_len - x[0].numel()), mode='constant', value=0) for x in masdata]
        else:
            masdata = [torch.ones((max_len, max_len)) for i in range(len(datafull))]
    return data, torch.tensor(labels).to(device), torch.stack(masdata).to(device)


# train model with parameters for given # of epochs
def train(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, device, clip_value=2):
    print("Total steps :", epochs*len(train_dataloader))
    
    # Run Training Loop
    for epoch in range(epochs):
        print("EPOCH ", epoch)
        model.train()
        for step, batch in enumerate(train_dataloader): 
            # extract data from loader
            batch_inputs, batch_labels, batch_masks = \
                               tuple(b.to(device) for b in batch)
            model.zero_grad()
            # run model, do train steps
            outputs = model(batch_inputs, batch_masks)
            loss = loss_function(outputs.squeeze(), 
                             batch_labels.squeeze())
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            # log loss
            if step%LOGSTEPS==0:
                print(loss)
        torch.save(model.state_dict(), "torchsaved/"+MODSTR+str(epoch)+".pt")  
    return model

# margin rank-based loss, ensures separate candidates 
# are correctly distant from eachother
vmask = (torch.triu(torch.ones(32, 32))*2-torch.ones(32, 32))*-1
vmask = vmask.to(device)
mse = nn.MSELoss()

# margin based rank loss, TODO sanity check again to make sure it's fine
def rank_loss(preds, golds):
    totloss = 0
    lim = int(len(preds)/RK_DIV)
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
    train(mod, optimizer, scheduler, loss, epochs, 
                  loader, device, clip_value=2)

if __name__ == "__main__":

    # TODO argumentize this stuff for applicability to other datasets
    xlm_tok = AutoTokenizer.from_pretrained('xlm-roberta-base')
    CAUSAL=True
    TSPLIT = 0.7
    LOGSTEPS = 500
    RK_DIV = 1
    LPAIR = "en_de"
    SCORE = "cqe"
    print("loading data")
    # load in data, split up
    xdata, ydata, padds = create_sortedbatch_data(LPAIR, SCORE, xlm_tok, 32)
    print("data loaded")
    newlen = int((len(xdata)*TSPLIT)/32)*32
    xtrain, ytrain, paddtrains = xdata[:newlen], ydata[:newlen], padds[:newlen]
    trainloader = DataLoader(RegressionDataset(xtrain, ytrain, paddtrains), batch_size=32, shuffle=False, collate_fn=collate_custom)
    # load in model
    model = XLMCometRegressor(drop_rate=0.1)
    model.load_state_dict(torch.load("./torchsaved/demsedone.pt"))
    # print("model loaded")
    # keep non-causal, train MSE, then see if we can get rank loss working
    #MODSTR = "demsecausalcoarse"
    #run_model_train_params(1e-4, 5, trainloader, model, mse)
    # MODSTR = "demsecausalfine"
    # run_model_train_params(1e-5, 5, trainloader, model, mse)
    MODSTR = "derlcausalfine"
    run_model_train_params(1e-5, 20, trainloader, model, rank_loss)





