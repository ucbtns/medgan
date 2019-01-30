"""
TRAIN GANOMALY
Author: Noor Sajid

"""
# LIBRARIES
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
os.chdir('E:\ganomly')
from options import Options
from lib.data import load_data
from lib.model import Ganomaly
from lib.evaluate import roc

##
# def main():
opt = Options().parse()
dataloader = load_data(opt)
model = Ganomaly(opt, dataloader)
torch.cuda.empty_cache() 
model.train()

# Test out the model: 
per, an, gt, dat = model.test()

def assessment(an, gt, dat):
    groundtruth = an.cpu().data.numpy()
    anomaly = gt.cpu().data.numpy()
    
    id_u = []
    for i in np.array(dat):
        id_u.append(i)
    
    id_s = []
    for i in id_u: 
        for j in i:
            j = j[-26:]
            id_s.append(j)
    
    df = pd.concat([pd.DataFrame(id_s), pd.DataFrame(groundtruth), pd.DataFrame(anomaly)], axis=1)
    df.columns = ['id', 'an', 'gt']
    return df

df = assessment(an, gt, dat)
outputp = Path(opt.outf + '/' + opt.name + '/')
df.to_csv(outputp +'/output_scores.csv') # saving the results

# save ROC curve from the test data:
#roc(df['gt'], df['an'], "E:\\ganomly\\256_output\\gano_1\\roc.png")

roc(df['gt'], df['an'], "roc.png")
#df[(df['gt'] == 0) & (df['an'] <= 0.1) ]

labels = df['gt']
scores = df['an']
fpr = dict()
tpr = dict()
roc_auc = dict()

# True/False Positive Rates.
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
import matplotlib.pyplot as plt
plt.plot(fpr)
plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

        plt.savefig(saveto)
        plt.close()


