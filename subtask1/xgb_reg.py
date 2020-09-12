import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
import logging
from sklearn.svm import SVR
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import wandb
from xgboost.sklearn import XGBRegressor


def remove_id(text):
    text = text.split('_', 1)[1]
    return text

#train_path_answers = 'public-data_ahmed/train_data_sets/psychopred_task1_train_answers.tsv'
train_path_answers = './psychopred_task1_train+dev_answers.tsv'
#train_path_students = 'public-data_ahmed/train_data_sets/psychopred_task1_train_students.tsv'
train_path_students = './psychopred_task1_train+dev_students.tsv'

# Testdaten
test_path_answers = 'public-data/test_data_sets/psychopred_task1_test_answers.tsv'

#dev_path_answers = 'public-data_ahmed/dev_data_sets/psychopred_task1_dev_answers.tsv'
#dev_path_answers = 'public-data_ahmed/dev_data_sets/psychopred_task1_dev_answers.tsv'
#dev_path_students = 'public-data_ahmed/dev_data_sets/psychopred_task1_dev_students.tsv'
#dev_path_ranks = 'public-data_ahmed/dev_data_sets/psychopred_task1_dev_student_ranks.tsv'



train_answers = pd.read_csv(train_path_answers, delimiter="\t",names=["student_ID","image_no","answer_no","UUID","MIX_text"],skiprows=1)
train_students = pd.read_csv(train_path_students, delimiter="\t")
#dev_answers = pd.read_csv(dev_path_answers, delimiter="\t")
dev_answers = pd.read_csv(test_path_answers, delimiter="\t")
#dev_students = pd.read_csv(dev_path_students, delimiter="\t")
#dev_ranks = pd.read_csv(dev_path_ranks, delimiter="\t")

print(train_answers.head())
print(dev_answers.head())

train_answers["UUID"] = train_answers["UUID"].apply(remove_id)
dev_answers["UUID"] = dev_answers["UUID"].apply(remove_id)

train_answers.drop(["image_no", "answer_no"], axis=1, inplace=True)
dev_answers.drop(["image_no", "answer_no"], axis=1, inplace=True)

train_new = train_answers.pivot(index='student_ID', columns='UUID', values='MIX_text').rename_axis(None, axis=1).reset_index()
dev_new = dev_answers.pivot(index='student_ID', columns='UUID', values='MIX_text').rename_axis(None, axis=1).reset_index()

train_new = pd.merge(train_new, train_students, on='student_ID')
#dev_new = pd.merge(dev_new, dev_students, on='student_ID')

cols = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2', '6_1', '6_2', '7_1', '7_2', '8_1', '8_2', '9_1', '9_2','10_1', '10_2', '11_1', '11_2', '12_1', '12_2', '13_1', '13_2', '14_1', '14_2', '15_1', '15_2']
train_new["text"] = train_new[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
train_new["text"] = train_new["text"].str.strip()
#train_new["text"] = train_new["text"].str.lower()
print(train_new.shape)
dev_new["text"] = dev_new[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
dev_new["text"] = dev_new["text"].str.strip()
#dev_new["text"] = dev_new["text"].str.lower()
train_new["combined"] = train_new["german_grade"] + train_new["english_grade"]+train_new["math_grade"]+train_new["lang_iq"]+train_new["logic_iq"]

#train = train_new.drop(["student_ID", "english_grade","math_grade","lang_iq","logic_iq"], axis=1, )
#dev = dev_new.drop(["student_ID", "english_grade","math_grade","lang_iq","logic_iq"], axis=1,)

train = pd.DataFrame(
    {'text': train_new["text"],
     'target': train_new["combined"]
    })

dev = pd.DataFrame(
    {'text': dev_new["text"]
    })
print(train.head())
print(dev.head())

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(train["text"])
dev_tfidf = tfidf.transform(dev["text"])


xgbr = XGBRegressor(learning_rate=0.01607,max_depth=6,min_child_weight=4,
                    n_estimators=490,nthread=10,subsample=0.7,colsample_bytree=0.7)
xgbr.fit(train_tfidf,train["target"])
pred = xgbr.predict(dev_tfidf)

res = pd.DataFrame(
    {'student_ID': dev_new["student_ID"],
     'target': pred
    })

final_df = res.sort_values(by=['target'], ascending=False)
ranks = []

for i in range(len(final_df)):
    ranks.append(i+1)

final_df["ranks"] = ranks

final_df.to_csv("reg_preds_final_comb.tsv",index=False, header=False,columns = ["student_ID","ranks"],sep='\t')


final_df = final_df.sort_values(by=['student_ID'], ascending=False)
#dev_ranks  = dev_ranks.sort_values(by=['student_ID'], ascending=False)


#scores = pearsonr(dev_ranks["rank"], final_df["ranks"])
#print(scores)                                                                                                                
