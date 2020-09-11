import logging
import torch
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy # accuracy_multilabel,
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, f1_score
from torch import Tensor
import numpy as np
labels = ['F2','M0','F4','L3','M1','05','F5','M3','A0','A4','F1','L0','L1','M5','F0','04','02','A2','M4','03','00','M2','L4','A5','F3','A1','A3','L5','L2']


#define metrics
def F1_macro(y_pred:Tensor, y_true:Tensor, average = 'macro', sample_weight = None):
    y_pred = np.argmax(y_pred.cpu(), axis = 1) 
    return f1_score(y_true.cpu(), y_pred.cpu(), average = average, sample_weight = sample_weight)

def F1_micro(y_pred:Tensor, y_true:Tensor):
    return F1_macro(y_pred.cpu(), y_true.cpu(), average = 'micro')

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

databunch = BertDataBunch('./', './',
                          tokenizer='bert-base-multilingual-cased',
                          train_file='train.csv',
                          val_file='dev.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          batch_size_per_gpu=64,
                          max_seq_length=128,
                          multi_gpu=False,
                          multi_label=False,
                          model_type='roberta')

device_cuda = torch.device('cuda')

metrics = []
metrics.append({'name': 'accuracy', 'function': accuracy})
metrics.append({'name': 'F1_macro', 'function': F1_macro})
metrics.append({'name': 'F1_micro', 'function': F1_micro})

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-multilingual-cased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir='./output/',
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=True,
						multi_label=False,
						logging_steps=500)

learner.fit(
		epochs=10,
		lr=6e-4,
		validate=True, 	# Evaluate the model after each epoch
		schedule_type="warmup_cosine",
		optimizer_type="lamb")

learner.validate()
learner.save_model()