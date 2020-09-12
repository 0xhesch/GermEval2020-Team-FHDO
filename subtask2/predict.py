#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import pandas as pd
from fast_bert.prediction import BertClassificationPredictor
import pickle
import json

MODEL_PATH = 'output/model_out/'

predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path='./', # location for labels.csv file
				multi_label=False,
				model_type='bert',
				do_lower_case=False)

#multi prediction
test_data = pd.read_csv('dev.csv')

x = 0
for item in test_data.text:
	prediction = predictor.predict(item)
	with open('predictions.tsv', 'a') as fp:
		#print(str(test_data.id[x]) + '\t' + prediction[0][0][0] + '\t' + prediction[0][0][1] + '\n')
		fp.write(str(test_data.id[x]) + '\t' + prediction[0][0][0] + '\t' + prediction[0][0][1] + '\n')
	x = x + 1
