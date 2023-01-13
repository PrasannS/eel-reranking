# Pipeline for training to get a causal model that matches some metric
import sys
sys.path.insert(1, '/mnt/data1/prasann/latticegen/lattice-generation/COMET')
from COMET.comet.models.regression.referenceless import ReferencelessRegression
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import csv
import sys
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm_
from generate_utils.process_traindata import create_sortedbatch_data
import random
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

csv.field_size_limit(sys.maxsize)

class XLMCometRegressorAvg(nn.Module):
    
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
    losssum = 0
    losstot = 0
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
            losssum+=loss
            losstot+=1
            #print("epoch done: ", losssum/losstot)
            # log loss
            if step%LOGSTEPS==0:
                #torch.cuda.empty_cache()
                print(losssum/losstot)
                losssum=0
                losstot=0
        
        if epoch%5==0 and CHECKPOINTS:
            torch.save(model.state_dict(), "torchsaved/"+MODSTR+str(epoch)+".pt")  
    return model

# margin rank-based loss, ensures separate candidates 
# are correctly distant from eachother
vmask = (torch.triu(torch.ones(4, 4))*2-torch.ones(4, 4))*-1
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

# shuffles up batches, but makes sure that stuff stays within size 4 clumps
def randomize_batches(xd, yd, pd):
    assert len(xd)==len(yd)
    indlist = list(range(int(len(xd)/4)))
    random.shuffle(indlist)
    xres, yres, pres = [], [], []
    for i in indlist:
        xres.extend(xd[i*4:(i+1)*4])
        yres.extend(yd[i*4:(i+1)*4])
        pres.extend(pd[i*4:(i+1)*4])
    assert len(xd) == len(xres)
    assert len(xres) == len(yres)
    assert len(pres) == len(yres)
    return xres, yres, pres



if __name__ == "__main__":
    TEST = True
    # TODO argumentize this stuff for applicability to other datasets
    xlm_tok = AutoTokenizer.from_pretrained('xlm-roberta-base')
    CAUSAL=True
    TSPLIT = 0.7
    LOGSTEPS = 500
    RK_DIV = 1
    # take up to 4 batches from each lattice, might help make training more stable?
    EX_SAMPLES = 8
    CHECKPOINTS = False
    LPAIR = "fr_en"
    SCORE = "bleurt"
    print("loading data")
    # load in data, split up
    xdata, ydata, padds = create_sortedbatch_data(LPAIR, SCORE, xlm_tok, 4, EX_SAMPLES)
    print("data loaded")
    newlen = int((len(xdata)*TSPLIT)/4)*4
    xtrain, ytrain, paddtrains = xdata[:newlen], ydata[:newlen], padds[:newlen]
    xtrain, ytrain, paddtrains = randomize_batches(xtrain, ytrain, paddtrains)
    # TODO change these back
    trainloader = DataLoader(RegressionDataset(xtrain, ytrain, paddtrains), batch_size=16, shuffle=True, collate_fn=collate_custom)
    # load in model
    model = XLMCometRegressorAvg(drop_rate=0.1)
    # model.load_state_dict(torch.load("./torchsaved/maskedcont4.pt"))
    # print("model loaded")
    # keep non-causal, train MSE, then see if we can get rank loss working
    #MODSTR = "demsecausalcoarse"
    #run_model_train_params(1e-4, 5, trainloader, model, mse)
    #MODSTR = "frenbleurtcourse"
    #run_model_train_params(1e-4, 5, trainloader, model, mse)
    #MODSTR = "flbleurtcoarse"
    RK_DIV = 1
    #run_model_train_params(5e-5, 10, trainloader, model, rank_loss)
    MODSTR = "flbleurtbigbatchmse"
    #run_model_train_params(1e-4, 6, trainloader, model, rank_loss)
    run_model_train_params(1e-4, 11, trainloader, model, mse)
    torch.save(model.state_dict(), "torchsaved/"+MODSTR+"done"+".pt")  

# Old XLMComet model without average normalization
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