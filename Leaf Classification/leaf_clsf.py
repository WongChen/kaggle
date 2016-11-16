import tensorflow as tf 
import numpy as np
import pandas as pd

def species2int(species):
	return label2int.get(species)


# loding data
df_train = pd.read_csv('data/train.csv')
del df_train['id']
COLUMNS = df_train.columns.values[1:].tolist()

ORIGINAL_LABEL = df_train['species'].values
# label
label_cln = df_train.groupby('species').species.all().values


# 2 dict to exchange
int2label = {k: label_cln[k] for k in range(label_cln.__len__	())}
label2int = {v:u for [u,v] in int2label.iteritems()}


LABEL_COLUMNS = "label"
int_label = map(species2int,ORIGINAL_LABEL)


CONTINUOUS_COLUMNS = df_train.columns.values[1:].tolist()

df_train[LABEL_COLUMNS] = int_label

def input_fn(df):
	continous_columns = {
			k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS
			}
	feature_columns = dict(continous_columns)
	label = tf.constant(df[LABEL_COLUMNS].values)
	return feature_columns,label


def int2species(index):
	return int2label.get(index)

def train_input_fn():
	return input_fn(df_train)

# all real columns
for feature in CONTINUOUS_COLUMNS:
	locals()[feature] = tf.contrib.layers.real_valued_column(feature)

# test set
df_test = pd.read_csv('data/test.csv')
del df_test['id']

def eval_input_fn(df):
	return dict({k: tf.constant(df_test[k].values,dtype=tf.float32) for k in CONTINUOUS_COLUMNS})
def test_input_fn():
	return eval_input_fn(df_test)
test = eval_input_fn(df_test)
model_dir = "./model"
m = tf.contrib.learn.LinearClassifier(feature_columns=[eval(feature) for feature in CONTINUOUS_COLUMNS],
		optimizer=tf.train.FtrlOptimizer(
			    learning_rate=0.1,
			    l1_regularization_strength=1.0,
			    l2_regularization_strength=1.0
			),model_dir=model_dir)
# comment to restore
m.fit(input_fn=train_input_fn, steps=10000)

#resluts = m.predict(x=eval_input_fn(df_test),as_iterable=False)
results = m.predict(input_fn=test_input_fn)
print results,results.__len__()
