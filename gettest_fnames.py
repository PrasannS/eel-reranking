import os
import pickle 
import csv
import pandas as pd

ende = "custom_output/data/mt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.904_0.0_0.9/"
fren = "custom_output/data/mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.904_0.0_0.9/"

if __name__ == "__main__":
    # load in stuff
    base = ende
    data = pd.read_csv("german_test.csv")
    fl = os.listdir(base)
    srcs = list(set(data['src']))
    usegraphs = []
    # go through and find everything where the src is same
    num = 0
    print(len(srcs))

    for name in fl:
        g = pickle.load(open(base+name, 'rb'))
        if g.document in srcs:
            usegraphs.append({
                'src':g.document,
                'fname':base+name
            })
        if num%100==0:
            print(num)
        num+=1
    print(len(usegraphs))
    ug = pd.DataFrame(usegraphs)
    ug.to_csv("german_fnames.csv")


    
