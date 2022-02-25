import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import metrics
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
	cluster_metric = 'hamming'
	cluster_method = 'hdbscan'
	use_weights = True
	n_particles = 10000 # for particle filtering, feature vects will be this long
	
	if cluster_method == 'dbscan':
		clusterer = DBSCAN(
			eps=.35, 
			min_samples=25,
			metric='precomputed' if use_weights else cluster_metric,
			n_jobs=-1
		)
		# 'jaccard', .5
		# 'hamming', .1, .3,
	elif cluster_method == 'hdbscan':# and not use_weights:
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=30,
			min_samples=5,
			metric=cluster_metric,
			core_dist_n_jobs=-1,
			# approx_min_span_tree=not use_weights
		)
		# 40, 20
	else:
		raise ValueError("Invalid cluster_method: {}".format(cluster_method))

	for start_pattern in starts_to_cluster:
		# subset data
		subset_data = subset_attr_data(
			tp_data,
			STR_starts == start_pattern
		)

		# Get position weights for clustering
		attrs = subset_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
		attr_mgm = np.max(np.abs(attrs), axis=1).mean(0)

		# get one hot seqs to cluster
		subset_strings = np.array([
			np.array(list(c[-n_per_side-str_pad_size:-str_pad_size])) 
				for c in subset_data['pre_strings']
		])

		if cluster_metric == 'hamming':
			X = subset_strings
			X[X == 'A'] = 0
			X[X == 'C'] = 1
			X[X == 'G'] = 2
			X[X == 'T'] = 3
			base_mapping = np.sort(np.unique(X))
			X = np.searchsorted(base_mapping, X)
		else:
			X = np.stack([
				(subset_strings == 'A').astype(int),
				(subset_strings == 'C').astype(int),
				(subset_strings == 'G').astype(int),
				(subset_strings == 'T').astype(int)
			]).transpose(1,0,2)
			X = X.reshape(X.shape[0], -1)

		# Cluster
		if use_weights and cluster_method == 'dbscan':
			# clusterer = hdbscan.HDBSCAN(
			# 	min_cluster_size=50,
			# 	min_samples=10,
			# 	cluster_selection_epsilon=.1,
			# 	metric=cluster_metric,
			# 	core_dist_n_jobs=-1,
			# 	# w=np.ones(n_per_side),
			# 	algorithm='generic',
			# 	approx_min_span_tree=False
			# )
			# clust_labels = clusterer.fit_predict(X)
			print('Calculating distance matrix...')
			dist_mat = metrics.pairwise_distances(
				X,
				metric=cluster_metric,
				n_jobs=-1,
				w=attr_mgm / attr_mgm.sum()
			)
			print("\tcomplete")
			clust_labels = clusterer.fit_predict(dist_mat)
		elif use_weights and cluster_method == 'hdbscan':
			"""
				Instead of normal weighting, coursely do it by expanding
				feature space such that each feature appears a proporsional 
				number of times to it weight. This isn't optimal, but HDBSCAN
				doesn't work properly with precomputed distances or a weight
				argument to the distance metric.
			"""
			particle_dist = np.round(attr_mgm / attr_mgm.sum() * n_particles).astype(int)
			expansion_inds = []
			for i in range(X.shape[1]):
				expansion_inds.extend([i] * particle_dist[i])
			
			X = X[:, expansion_inds]
			clust_labels = clusterer.fit_predict(X)
		else:
			clust_labels = clusterer.fit_predict(X)
		print(*zip(*np.unique(clust_labels, return_counts=True)))

		# Get position probability matrix for each cluster
		sorted_labels = np.sort(np.unique(clust_labels))
		cluster_ppms = []

		fig, axs = plt.subplots(len(sorted_labels), 2, dpi=150, squeeze=False)
		fig.suptitle("Logo before {}".format(start_pattern))
		
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
			logomaker.Logo(mean_attr_df, ax=axs[i, 0])

			# get counts
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

		break