import numpy as np
import networkx as nx
import pandas as pd
import math
import random
from statistics import mean
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

def evaluate_link_prediction(embedding_train, nx_G):
	WINDOW = 1 # Node2Vec fit window
	MIN_COUNT = 1 # Node2Vec min. count
	BATCH_WORDS = 4 # Node2Vec batch words
	max_iter = 2000

	mdl = embedding_train

	# create embeddings dataframe
	emb_df = (
			pd.DataFrame(
					[mdl.wv.get_vector(str(n)) for n in nx_G.nodes()],
					index = nx_G.nodes
			)
	)

	print(emb_df.head())
	#Create Training Data
	unique_nodes = list(nx_G.nodes())
	#all_possible_edges = [(x,y) for (x,y) in product(unique_nodes, unique_nodes)]
	#Genrate random false edges instead of product
	all_possible_edges = gen_rand_edges(nx_G.edges(), unique_nodes)
	# generate edge features for all pairs of nodes
	edge_features = [
		(mdl.wv.get_vector(str(i)) + mdl.wv.get_vector(str(j))) for i,j in all_possible_edges
	]

	# get current edges in the network
	edges = list(nx_G.edges())

	# create target list, 1 if the pair exists in the network, 0 otherwise
	is_con = [1 if e in edges else 0 for e in all_possible_edges]

	print(sum(is_con))
	X = np.array(edge_features)
	y = is_con

	# train test split
	x_train, x_test, y_train, y_test = train_test_split(X,	y,	test_size = 0.3	)
	rfc_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
	clf = Pipeline(steps=[("sc", StandardScaler()), ("clf", rfc_clf)])
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	y_true = y_test

	
	value = eval_report(clf, x_train, y_train, x_test, y_test)
	return value

def gen_rand_edges(teCt, unique_nodes):
	num = 3*len(teCt)
	seen = set(teCt)
	x, y = random.choice(unique_nodes), random.choice(unique_nodes)
	t = 0
	while t < num:
		seen.add((x, y))
		t = t + 1
		x, y = random.choice(unique_nodes), random.choice(unique_nodes)
		while (x, y) in seen or (y, x) in seen:
			x, y = random.choice(unique_nodes), random.choice(unique_nodes)
	
	seen = list(seen)
	return seen

def eval_report(model,x_train, y_train, x_test, y_test):
	test_pred = model.predict(x_test)
	test_acc = accuracy_score(y_test, test_pred)
	prec_per, recall_per, threshold_per = precision_recall_curve(y_test, test_pred)
	prec_per = prec_per[::-1]
	recall_per = recall_per[::-1]
	aupr_value = np.trapz(prec_per, x=recall_per)
	avg_prec_value = average_precision_score(y_test, test_pred)
	AUC = roc_auc_score(y_test, test_pred)
	test_pred_label = np.copy(test_pred)
	a = np.mean(test_pred_label)

	for i in range(len(test_pred)):
		if test_pred[i] < a:
			test_pred_label[i] = 0
		else:
			test_pred_label[i] = 1
	acc_score_value = accuracy_score(y_test, test_pred_label)
	bal_acc_score_value = balanced_accuracy_score(y_test, test_pred_label)
	f1_value = f1_score(y_test, test_pred_label)
	Recall = recall_score(y_test, test_pred_label)
	Precision = precision_score(y_test, test_pred_label)

	print('\nTest accuracy:' + str(test_acc) + "\nAUC:" + str(AUC) + "\nPrecision:" + str(Precision) +
"\nRecall:" + str(Recall) + "\nAUPR:" + str(aupr_value) + "\nAvgPrecision" + str(avg_prec_value) +
"\nAccScore:" + str(acc_score_value) + "\nBalAccScore:" + str(bal_acc_score_value) + "\nF1:" + str(f1_value))
	return [test_acc, AUC, Precision, Recall, aupr_value, avg_prec_value, acc_score_value, bal_acc_score_value, f1_value]

