# Pipeline for training to get a causal model that matches some metric
import sys
sys.path.insert(1, '/mnt/data1/prasann/latticegen/lattice-generation/COMET')
from tfr_models.comet.models.regression.referenceless import ReferencelessRegression
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import csv
import sys
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm_
from generate_utils.process_traindata import create_sortedbatch_data
import random
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

csv.field_size_limit(sys.maxsize)

# TODO modifications to make to get the format to fit
# - Change initial tokenization code
# - Change data loader, store HYP / SRC separately, include masks
# - Change forward pass a little
# - Extract score from dict returned

class RegressionDataset(Dataset):
    def __init__(self, srcs, mts, srcmasks, mtmasks, labels):
        assert len(srcs) == len(labels)
        self.mts = mts
        self.srcs = srcs
        self.labels = labels

    def __getitem__(self, i):
        return self.mts[i], self.srcs[i], self.labels[i]

    def __len__(self):
        return len(self.mts)

# process data for a single batch
def collate_custom(datafull):
    # Extract out data from input tuples
    with torch.no_grad():
        mts = [d[0] for d in datafull]
        srcs = [d[1] for d in datafull]
        labels = [d[2] for d in datafull]
        tokmt = xlm_tok(mts, device=device)
        toksrc = xlm_tok(srcs, device=device)

    return tokmt, toksrc, labels


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
            batch_mts, batch_srcs, batch_labels = \
                               tuple(b.to(device) for b in batch)
            model.zero_grad()
            # run model, do train steps
            outputs = model(
                src_input_ids=batch_srcs.input_ids, 
                src_attention_mask=batch_srcs.attention_mask,
                mt_input_ids=batch_mts.input_ids,
                mt_attention_mask=batch_mts.attention_mask)['score']
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
    LPAIR = "en_de"
    SCORE = "comet"
    print("loading data")
    # load in data, split up
    srcs, mts, labels = create_sortedbatch_data(LPAIR, SCORE, xlm_tok, 4, EX_SAMPLES)
    print("data loaded")
    newlen = int((len(srcs)*TSPLIT)/4)*4
    srcs, mts, labels = srcs[:newlen], mts[:newlen], labels[:newlen]
    srcs, mts, labels = randomize_batches(srcs, mts labels)
    # TODO change these back
    # srcs, mts, srcmasks, mtmasks, labels
    trainloader = DataLoader(RegressionDataset(srcs, mts, labels), batch_size=4, shuffle=False, collate_fn=collate_custom)
    # load in model
    model = ReferencelessRegression()
    model.load_from_checkpoint("COMET/lightning_logs/version_30/checkpoints/epoch=9-step=130000.ckpt").to(device)
   
    RK_DIV = 1
    #run_model_train_params(5e-5, 10, trainloader, model, rank_loss)
    MODSTR = "endecometrankloss"
    #run_model_train_params(1e-4, 6, trainloader, model, rank_loss)
    run_model_train_params(1e-5, 2, trainloader, model, mse)
    run_model_train_params(1e-5, 10, trainloader, model, rank_loss)

    torch.save(model.state_dict(), "torchsaved/"+MODSTR+"done"+".pt")  
