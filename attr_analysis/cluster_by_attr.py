import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
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
	str_motif_len = 2
	str_pad_size = 6

	n_per_side = 20

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

	tp_data = subset_attr_data(
		attr_data, 
		(attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
	)
	tn_data = subset_attr_data(
		attr_data, 
		(attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
	)

	# Add data on start/end of STR, since clustering will be done 
	#	independantly for each type
	min_count = 100 # must be at least this many examples to cluster for STR start/end

	STR_starts = np.array([
		s[-str_pad_size : -str_pad_size+str_motif_len] for s in tp_data['pre_strings']
	])
	STR_ends = np.array([
		s[str_pad_size-str_motif_len : str_pad_size] for s in tp_data['post_strings']
	])

	print(*zip(*np.unique(STR_starts, return_counts=True)))
	print(*zip(*np.unique(STR_ends, return_counts=True)))

	starts_to_cluster = [
		s for s,c in zip(*np.unique(STR_starts, return_counts=True)) if c >= min_count
	]
	ends_to_cluster = [
		s for s,c in zip(*np.unique(STR_ends, return_counts=True)) if c >= min_count
	]

	# For each STR start pattern subset, cluster seqs by attribution weights
	cluster_metric = 'l1'
	cluster_method = 'hdbscan'
	use_weights = False
	
	if cluster_method == 'dbscan':
		clusterer = DBSCAN(
			eps=.2, 
			min_samples=40,
			metric=cluster_metric,
			n_jobs=-1
		)
	elif cluster_method == 'hdbscan':# and not use_weights:
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=25,
			min_samples=4,
			metric=cluster_metric,
			core_dist_n_jobs=-1
		)
	else:
		raise ValueError("Invalid cluster_method: {}".format(cluster_method))

	for start_pattern in starts_to_cluster:
		print(start_pattern)

		# subset data
		subset_data = subset_attr_data(
			tp_data,
			STR_starts == start_pattern
		)

		# get data in useful format and cluster
		X = subset_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
		X = X.reshape(X.shape[0], -1)

		clust_labels = clusterer.fit_predict(X)
		print(*zip(*np.unique(clust_labels, return_counts=True)))

		# Get position probability matrix for each cluster
		sorted_labels = np.sort(np.unique(clust_labels))
		cluster_ppms = []

		plt_height = 1.0
		fig_height = len(sorted_labels) * plt_height
		fig, axs = plt.subplots(
			len(sorted_labels), 2, figsize=(12.8,fig_height), squeeze=False,
			sharex=True, sharey='col'
		)
		# if len(sorted_labels) == 1:
		# 	axs = [axs]
		fig.suptitle("Logo before {}".format(start_pattern))

		max_attr_mag = 0
		min_attr_mag = 0
		
		for i,cluster in enumerate(sorted_labels):
			clust_mask = clust_labels == cluster
			clust_strings = np.array(
				[np.array(list(c[-n_per_side-str_pad_size:-str_pad_size])) for c in subset_data['pre_strings'][clust_mask]]
			)

			# get mean attributions
			mean_ignore_0 = True
			if mean_ignore_0:
				full_attrs = subset_data['pre'][clust_mask][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
				full_attrs[full_attrs == 0] = np.nan
				with np.errstate(all='ignore'):
					mean_attrs = np.nan_to_num(np.nanmean(full_attrs, 0))
			else:
				mean_attr = subset_data['pre'][clust_mask][
					:,:4:,-n_per_side-str_pad_size:-str_pad_size].mean(0)
			mean_attr_df = pd.DataFrame(
				mean_attrs.T, 
				columns=['A', 'C', 'G', 'T']
			)

			# For scaling plot
			max_attr_mag = max(
				max_attr_mag, 
				np.where(mean_attr_df.values > 0, mean_attr_df.values, 0).sum(1).max()
			)
			min_attr_mag = min(
				min_attr_mag, 
				np.where(mean_attr_df.values < 0, mean_attr_df.values, 0).sum(1).min()
			)

			logomaker.Logo(mean_attr_df, ax=axs[i, 0])
			axs[i, 0].set_ybound(min_attr_mag, max_attr_mag)

			# get position probability matrix
			counts_mat = np.stack([
				(clust_strings == 'A').sum(0),
				(clust_strings == 'C').sum(0),
				(clust_strings == 'G').sum(0),
				(clust_strings == 'T').sum(0)
			])
			ppm = counts_mat / counts_mat.sum(0, keepdims=True)
			ppm_df = pd.DataFrame(ppm.T, columns=['A', 'C', 'G', 'T'])

			# logo = logomaker.Logo(ppm_df, stack_order='fixed')
			# plt.show()
			logo = logomaker.Logo(ppm_df, ax=axs[i, 1])
			# plt.show()

		# Format plot
		plt.setp(
			axs, 
			xticks=np.array(list(range(0, ppm.shape[1], 5))), 
			xticklabels=list(range(-ppm.shape[1], 0, 5))
		)
		plt.tight_layout(pad=.3, w_pad=.3, h_pad=.1)

		# # plot prediction confidence dists by cluster
		# conf_df = pd.DataFrame({
		# 	'pred_conf': subset_data['predictions'],
		# 	'cluster': clust_labels
		# })
		# fig, axs = plt.subplots(1, 2, figsize=(6.4,fig_height))
		# sns.violinplot(
		# 	x='pred_conf',
		# 	y='cluster',
		# 	orient='h',
		# 	scale='count',
		# 	order=sorted_labels,
		# 	cut=0,
		# 	data=conf_df,
		# 	ax=axs[0]
		# )

		# sns.countplot(
		# 	y='cluster',
		# 	data=conf_df,
		# 	# order=conf_df['cluster'].value_counts(ascending=False).index,
		# 	order=sorted_labels,
		# 	ax=axs[1]
		# )
		# # axs[1].set_xscale('log')
		# axs[1].set_xlabel('Number of examples')
		# axs[1].set_ylabel(None)

		# abs_values = conf_df['cluster'].value_counts(ascending=False)
		# rel_values = conf_df['cluster'].value_counts(ascending=False, normalize=True) * 100
		# count_labels = [
		# 	f'{abs_values[l]} ({rel_values[l]:.1f}%)' for l in sorted_labels
		# ]
		# axs[1].bar_label(container=axs[1].containers[0], labels=count_labels)

		# plt.show()

		# plot prediction confidence dists by cluster
		conf_df = pd.DataFrame({
			'pred_conf': subset_data['predictions'],
			'cluster': clust_labels,
			'heterozygosity_score': subset_data['sample_data'].label.values
		})
		fig, axs = plt.subplots(1, 3)
		sns.violinplot(
			x='pred_conf',
			y='cluster',
			orient='h',
			scale='count',
			order=sorted_labels,
			cut=0,
			data=conf_df,
			ax=axs[0]
		)
		
		sns.violinplot(
			x='heterozygosity_score',
			y='cluster',
			orient='h',
			order=sorted_labels,
			cut=0,
			data=conf_df,
			ax=axs[1]
		)

		sns.countplot(
			y='cluster',
			data=conf_df,
			order=sorted_labels,
			ax=axs[2]
		)
		# axs[1].set_xscale('log')
		axs[2].set_xlabel('Number of examples')
		axs[2].set_ylabel(None)

		abs_values = conf_df['cluster'].value_counts(ascending=False)
		rel_values = conf_df['cluster'].value_counts(ascending=False, normalize=True) * 100
		count_labels = [
			f'{abs_values[l]} ({rel_values[l]:.1f}%)' for l in sorted_labels
		]
		axs[2].bar_label(container=axs[2].containers[0], labels=count_labels)

		plt.show()