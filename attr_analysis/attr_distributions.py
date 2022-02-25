"""Cluster by attributions in a way that allows for comparison of clusters 
by label or TP/TN status.
"""

import os
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, dbscan
import hdbscan

import logomaker


def reformat_attribution_pkl(attr_data):
	"""Reformat attribution data from as read from pkl to numpy arrays
	or pandas dataframes.
	"""
	attr_data['pre'] = np.stack(attr_data['pre'])
	attr_data['post'] = np.stack(attr_data['post'])
	attr_data['pre_strings'] = np.stack(attr_data['pre_strings'])
	attr_data['post_strings'] = np.stack(attr_data['post_strings'])

	# fix a typo
	if 'predicitons' in attr_data.keys():
		attr_data['predictions'] = attr_data.pop('predicitons')

	attr_data['predictions'] = np.array(attr_data['predictions'])
	attr_data['labels'] = np.array(attr_data['labels'])
	attr_data['sample_data'] = pd.DataFrame(attr_data['sample_data'])

	return attr_data


def add_prediction_features(attr_data, cutoff=.5):
	"""Add binary prediction features and their correctness to attr_data."""
	attr_data['binary_pred'] = (attr_data['predictions'] > cutoff).astype(int)
	attr_data['correct_pred'] = (attr_data['binary_pred'] == attr_data['labels'])
	attr_data['true_pos'] = (attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
	attr_data['true_neg'] = (attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
	return attr_data


def subset_attr_data(attr_data, subset_mask):
	attr_data = copy.deepcopy(attr_data)

	for k,v in attr_data.items():
		if isinstance(v, np.ndarray):
			attr_data[k] = v[subset_mask]
		elif isinstance(v, pd.DataFrame):
			attr_data[k] = v.loc[subset_mask]

	return attr_data



if __name__ == '__main__':
	"""Args:

		str_motif_len: length of STR motif(s) (e.g. 2 for {'CA', 'TG', 
			'GA'} motifs)
		str_pad_size: number of positions of attributions that are STR padding.
		n_per_side: number of loci to use in clustering.
		use_TP_TN: whether to use TP/TN status (True) or label (False) when
			seperating groups before clustering.
	"""
	str_motif_len = 2
	str_pad_size = 6

	n_per_side = 20

	use_TP_TN = True	

	save_plots = True
	show_plots = False

	# Load data
	motif_type = ['T', 'AC', '5comb'][2]

	# Select model's output dir
	if motif_type == 'AC':
		trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1'
	elif motif_type == 'T':
		trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1_T'
	elif motif_type == '5comb':
		trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1_AC-AG-AT-CT-GT'
	else:
		raise ValueError("Invalid motif_type: {}".format(motif_type))
	model_dir = 'version_4'

	# Load single attr type
	all_splits = False
	if all_splits:
		attr_file = 'ig_global_train_val_test.pkl'
	else:
		attr_file = 'ig_global.pkl'
	attr_data = pd.read_pickle(
		os.path.join(trained_res_dir, model_dir, 'attributions', attr_file)
	)

	# Reformat data and add prediction features
	attr_data = reformat_attribution_pkl(attr_data)
	attr_data = add_prediction_features(attr_data)
	attr_data['sample_data']['alpha_motif'] = attr_data['sample_data'].motif.apply(
		lambda x: ''.join(sorted([*x])) # motif name with letters sorted alphabetically
	)

	if use_TP_TN:
		pos_class_data = subset_attr_data(
			attr_data, 
			(attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
		)
		neg_class_data = subset_attr_data(
			attr_data, 
			(attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
		)
	else:
		pos_class_data = subset_attr_data(
			attr_data, 
			(attr_data['labels'] == 1)
		)
		neg_class_data = subset_attr_data(
			attr_data, 
			(attr_data['labels'] == 0)
		)

	# Add data on start/end of STR, since clustering will be done 
	#	independantly for each type
	min_count = 100 # must be at least this many examples to cluster for STR start/end

	pos_STR_starts = np.array([
		s[-str_pad_size : -str_pad_size+str_motif_len] for s in pos_class_data['pre_strings']
	])
	pos_STR_ends = np.array([
		s[str_pad_size-str_motif_len : str_pad_size] for s in pos_class_data['post_strings']
	])
	neg_STR_starts = np.array([
		s[-str_pad_size : -str_pad_size+str_motif_len] for s in neg_class_data['pre_strings']
	])
	neg_STR_ends = np.array([
		s[str_pad_size-str_motif_len : str_pad_size] for s in neg_class_data['post_strings']
	])

	print("Positive class data: {} examples".format(len(pos_class_data['sample_data'])))
	print(*zip(*np.unique(pos_STR_starts, return_counts=True)))
	print(*zip(*np.unique(pos_STR_ends, return_counts=True)))

	print("Negative class data: {} examples".format(len(neg_class_data['sample_data'])))
	print(*zip(*np.unique(neg_STR_starts, return_counts=True)))
	print(*zip(*np.unique(neg_STR_ends, return_counts=True)))

	pos_starts_to_cluster = {
		s for s,c in zip(*np.unique(pos_STR_starts, return_counts=True)) if c >= min_count
	}
	neg_starts_to_cluster = {
		s for s,c in zip(*np.unique(neg_STR_starts, return_counts=True)) if c >= min_count
	}
	starts_to_cluster = pos_starts_to_cluster & neg_starts_to_cluster
	# ends_to_cluster = [
	# 	s for s,c in zip(*np.unique(STR_ends, return_counts=True)) if c >= min_count
	# ]

	X_all = attr_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
	X_all = X_all.reshape(X_all.shape[0], -1)
	sns.displot(X_all.flatten())

	# Attr clipping
	clip_min_perc, clip_max_perc = 5, 95
	cutoffs = np.percentile(X_all, [clip_min_perc, clip_max_perc])
	X_all_clipped = np.clip(X_all, cutoffs[0], cutoffs[1])

	sns.displot(X_all_clipped.flatten())
	plt.show()



	for start_pattern in starts_to_cluster:
		print(start_pattern)

		# subset data
		pos_subset_data = subset_attr_data(
			pos_class_data,
			pos_STR_starts == start_pattern
		)
		neg_subset_data = subset_attr_data(
			neg_class_data,
			neg_STR_starts == start_pattern
		)

		X_pos = pos_subset_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
		X_pos = X_pos.reshape(X_pos.shape[0], -1)

		sns.displot(X_pos.flatten())
		sns.displot(np.clip(X_pos, cutoffs[0], cutoffs[1]).flatten())
		plt.show()		

		X_neg = pos_subset_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
		X_neg = X_neg.reshape(X_neg.shape[0], -1)

		sns.displot(X_neg.flatten())
		sns.displot(np.clip(X_neg, cutoffs[0], cutoffs[1]).flatten())
		plt.show()		

		break