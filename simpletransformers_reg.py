from simpletransformers.classification import ClassificationModel
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
import logging
import wandb

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)



def remove_id(text):
    text = text.split('_', 1)[1]
    return text


train_path_answers = 'public-data/train_data_sets/psychopred_task1_train_answers.tsv'
train_path_students = 'public-data/train_data_sets/psychopred_task1_train_students.tsv'
dev_path_answers = 'public-data/dev_data_sets/psychopred_task1_dev_answers.tsv'
dev_path_students = 'public-data/dev_data_sets/psychopred_task1_dev_students.tsv'
dev_path_ranks = 'public-data/dev_data_sets/psychopred_task1_dev_student_ranks.tsv'



train_answers = pd.read_csv(train_path_answers, delimiter="\t")
train_students = pd.read_csv(train_path_students, delimiter="\t")
dev_answers = pd.read_csv(dev_path_answers, delimiter="\t")
dev_students = pd.read_csv(dev_path_students, delimiter="\t")
dev_ranks = pd.read_csv(dev_path_ranks, delimiter="\t")


train_answers["UUID"] = train_answers["UUID"].apply(remove_id)
dev_answers["UUID"] = dev_answers["UUID"].apply(remove_id)




train_answers.drop(["image_no", "answer_no"], axis=1, inplace=True)
dev_answers.drop(["image_no", "answer_no"], axis=1, inplace=True)


train_new = train_answers.pivot(index='student_ID', columns='UUID', values='MIX_text').rename_axis(None, axis=1).reset_index()
dev_new = dev_answers.pivot(index='student_ID', columns='UUID', values='MIX_text').rename_axis(None, axis=1).reset_index()


train_new = pd.merge(train_new, train_students, on='student_ID')
dev_new = pd.merge(dev_new, dev_students, on='student_ID')


cols = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2', '6_1', '6_2', '7_1', '7_2', '8_1', '8_2', '9_1', '9_2','10_1', '10_2', '11_1', '11_2', '12_1', '12_2', '13_1', '13_2', '14_1', '14_2', '15_1', '15_2']
train_new["text"] = train_new[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
train_new["text"] = train_new["text"].str.strip()
train_new["text"] = train_new["text"].str.lower()
dev_new["text"] = dev_new[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
dev_new["text"] = dev_new["text"].str.strip()
dev_new["text"] = dev_new["text"].str.lower()


train = pd.DataFrame(
    {'text': train_new["text"],
     'label': train_new["logic_iq"]
    })

dev = pd.DataFrame(
    {'text': dev_new["text"],
     'label': dev_new["logic_iq"]
    })


train_full = pd.concat([train,dev])
print(train.head())
print(dev.head())


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

from scipy.stats import pearsonr, spearmanr


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]
eval_metrics = {
#
    "pearson":pearson_corr,
    "spearman":spearman_corr
  #  "f1_germ":f1_germ
  #  "roc_auc": sklearn.metrics.roc_auc_score,
 #   "avg_prc": sklearn.metrics.average_precision_score,

}


train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs':10,
    "learning_rate": 4e-5,
    "evaluate_during_training_steps": 50,
    'train_batch_size':16,
    'output_dir':'output_reg',
    'regression': True,
    "manual_seed": 42,
    'evaluate_during_training':False,
    "eval_batch_size": 16,
    "max_seq_length": 512,
    'wandb_project': 'reg_dbmdz_uncased_math'
}

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-german-dbmdz-uncased', num_labels=1, use_cuda=True, cuda_device=0, args=train_args)


# Train the model
model.train_model(train, eval_df=dev,**eval_metrics)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dev)
print(result)
#dev.drop(["german_grade"],axis=1, inplace=True)

pred, raw = model.predict(dev["text"])

print(pred)
#exit()
res = pd.DataFrame(
    {'student_ID': dev_new["student_ID"],
     'target': pred
    })
final_df = res.sort_values(by=['target'], ascending=False)
ranks = []
for i in range(len(final_df)):
    ranks.append(i+1)

final_df["ranks"] = ranks

final_df.to_csv("reg_preds_test.tsv",index=False, header=False,columns = ["student_ID","ranks"],sep='\t')


final_df = final_df.sort_values(by=['student_ID'], ascending=False)
dev_ranks  = dev_ranks.sort_values(by=['student_ID'], ascending=False)


scores = pearsonr(dev_ranks["rank"], final_df["ranks"])
print(scores)
wandb.log({"pearson_germ": scores[0],
            "p-value": scores[1]})

