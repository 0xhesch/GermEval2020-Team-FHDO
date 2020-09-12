import os, sys
from collections import defaultdict

#define paths
TRAIN_TEXT_PATH = './train/psychopred_task2_train_texts.tsv'
TRAIN_LABEL_PATH = './train/psychopred_task2_train_labels.tsv'

DEV_TEXT_PATH = './dev/psychopred_task2_dev_texts.tsv'
DEV_LABEL_PATH = './dev/psychopred_task2_dev_labels.tsv'

#define na's
NA='\\\\N\\\\N'

#read data
with open(TRAIN_TEXT_PATH, errors='ignore') as f:
    train_texts = f.readlines()
    
with open(TRAIN_LABEL_PATH, errors='ignore') as f:
    train_labels = f.readlines()
    
with open(DEV_TEXT_PATH, errors='ignore') as f:
    dev_texts = f.readlines()
    
with open(DEV_LABEL_PATH, errors='ignore') as f:
    dev_labels = f.readlines()
    
#add training header
train_csv = []
train_csv.append("id,text,label")

#add dev header
dev_csv = []
dev_csv.append("id,text,label")

#label list
labels = []

#build train csv for bert databunch
for i, entry in enumerate(train_texts):
    #ignore header
    if i == 0:
        continue
    text = entry.split('\t')[1].replace('"', '').rstrip()
    label = train_labels[i].split('\t')[1] + train_labels[i].split('\t')[2].rstrip()
    
    #drop na and fix M4^1
    if(label == NA):
        continue;
    if(label == 'M4^'):
        label = 'M4'
    train_csv.append(train_labels[i].split('\t')[0] + ',' + '"' + text + '"' + ',' + label)
    
    #add label to list, save distinct labels later
    labels.append(label)
    
#build dev csv for bert databunch
for i, entry in enumerate(dev_texts):
    #ignore header
    if i == 0:
        continue
    text = entry.split('\t')[1].replace('"', '').rstrip()
    label = dev_labels[i].split('\t')[1] + dev_labels[i].split('\t')[2].rstrip()
    
    #drop na and fix M4^1
    if(label == NA):
        continue;
    if(label == 'M4^'):
        label = 'M4'
    dev_csv.append(dev_labels[i].split('\t')[0] + ',' + '"' + text + '"' + ',' + label)
    
    #add label to list, save distinct labels later
    labels.append(label)
    
#save train and dev csv
with open("train.csv", "w") as out_train:
    out_train.write("\n".join(train_csv))
    
with open("dev.csv", "w") as out_dev:
    out_dev.write("\n".join(dev_csv))
    
#save distinct labelset
distinct_labels = list(set(labels))
with open("labels.csv", "w") as out_dev:
    out_dev.write("\n".join(distinct_labels))
