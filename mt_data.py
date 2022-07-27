import csv
import pandas as pd
import random
from os.path import exists
import sys

csv.field_size_limit(sys.maxsize)

# store dataset in file to ensure consistency (bc of shuffling)
def load_generate_set(num, dataset):
    fname = './translation_data/processed/'+dataset+"_"+str(num)+".csv"
    if exists(fname):
        dset = pd.read_csv(fname, index_col=False)
    else:
        dset = get_sample_set(num, dataset)
        dset.to_csv(fname)
    return dset


def get_sample_set (num, dataset):
    #TODO there's definetely a way to clean this up
    if dataset == 'en_de':
        with open("translation_data/news-commentary-v15.de-en.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            i = 0
            res = []
            # generate data for everything, shuffle and sample later
            for f in tsv_file:
                if i == 100000:
                    break
                res.append(f)
                i = i+1
        random.shuffle(res)
        res = res[:num]
        tmpdf = pd.DataFrame(res)
        tmpdf['de'] = tmpdf[0]
        tmpdf['en'] = tmpdf[1]
        del tmpdf[0]
        del tmpdf[1]
        return tmpdf
    elif dataset == 'fr_en':
        fr_raw = []
        en_raw = []
        with open('translation_data/news-commentary-v9.fr-en.fr') as f:
            for i in range(0, 8000):
                fr_raw.append(f.readline())
            #fr_raw.remove("")
        with open('translation_data/news-commentary-v9.fr-en.en') as f:
            for i in range(0, 8000):
                en_raw.append(f.readline())
            #en_raw.remove("")
        

        res = []
        for fr, en in zip(fr_raw, en_raw):
            tmp = {}
            tmp['fr'] = fr
            tmp['en'] = en
            res.append(tmp)
        random.shuffle(res)
        res = res[:num]
        tmpdf = pd.DataFrame(res)
        return tmpdf
    


