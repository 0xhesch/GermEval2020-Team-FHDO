import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
from typing import List
src = 'de'  # source language
trg = 'en'  # target language

train_path = './psychopred_task2_train.tsv'
dev_path = './psychopred_task2_dev.tsv'


train = pd.read_csv(train_path, delimiter='\t')
print(train.shape)
dev = pd.read_csv(dev_path, delimiter='\t')

#### Correcting a typo
train.loc[train['class'] == 'M4^', 'class'] = "M4"
train.loc[train['level'] == '4^', 'level'] = "4"
#### Removing NaNs
train.drop(train[train['class'] == '\\\\N\\\\N'].index, inplace=True)
print(train.shape)
results = []


#sample_text = train["OMT_text"]#.head(10)
#sample_text = "Das hier ist nur ein Test"
mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
model = MarianMTModel.from_pretrained(mname)
tok = MarianTokenizer.from_pretrained(mname)
counter = 0
for i in train["OMT_text"]:
    words = {}
    batch = tok.prepare_translation_batch(src_texts=[i])  # don't need tgt_text for inference
    gen = model.generate(**batch)  # for forward pass: model(**batch)
    words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
#print(words)
#for i,k in zip(words,sample_text):
#    print(k + ':\t' + i)
##äresults = []
    #for i in words:
    print(counter)
    counter +=1
    results.append(words[0])

train["EN"] = results
train.to_csv('train_with_translation.tsv',index=False, encoding='utf-8', sep='\t')
