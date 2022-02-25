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


def plot_cluster_attn_and_ppm(clust_labels, subset_data, n_per_side,
		str_pad_size, plt_height=1.0, label_desc=''):
	"""Plot cluster attribution and ppm for each cluster.

	TODO: handle post STR seqs
	
	Args:
		clust_labels: array of cluster labels
		subset_data: attribution data in the normal dict form
		n_per_side: number of samples per side STR used
		str_pad_size: size of STR padding
		plt_height: height of total plot is plt_height * num_clusters
	"""
	sorted_labels = np.sort(np.unique(clust_labels))

	fig, axs = plt.subplots(len(sorted_labels), 2, 
		figsize=(10,len(sorted_labels) * plt_height), 
		squeeze=False,
		sharex=True, 
		sharey='col'
	)

	fig.suptitle("Logo before {} {}".format(start_pattern, label_desc))

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

		logomaker.Logo(ppm_df, ax=axs[i, 1])

	# Format plot
	plt.setp(
		axs, 
		xticks=np.array(list(range(0, ppm.shape[1], 5))), 
		xticklabels=list(range(-ppm.shape[1], 0, 5))
	)
	# plt.tight_layout(pad=.3, w_pad=.3, h_pad=.1)

	return fig, axs, sorted_labels


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

	use_TP_TN = False	

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
	all_splits = True
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

	# For each STR start pattern subset, cluster seqs by attribution weights
	cluster_metric = 'hamming'
	cluster_method = 'hdbscan'
	dbscan_params = {
		'eps': .35,
		'min_samples': 25,
	}
	hdbscan_params = {
		'min_cluster_size': 30,
		'min_samples': 5,
	}
	use_weights = True
	# for particle filtering, feature vects will be about this long
	n_particles = 1000

	if cluster_method == 'dbscan':
		clusterer = DBSCAN(
			eps=dbscan_params['eps'],
			min_samples=dbscan_params['min_samples'],
			metric='precomputed' if use_weights else cluster_metric,
			n_jobs=-1
		)
		# 'jaccard', .5
		# 'hamming', .1, .3,
	elif cluster_method == 'hdbscan':# and not use_weights:
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=hdbscan_params['min_cluster_size'],
			min_samples=hdbscan_params['min_samples'],
			metric=cluster_metric,
			core_dist_n_jobs=-1,
			# approx_min_span_tree=not use_weights
		)
		# 40, 20
	else:
		raise ValueError("Invalid cluster_method: {}".format(cluster_method))

	# Create dir to save cluster results
	if save_plots:
		cluster_res_dir = '_'.join([
			motif_type, 
			model_dir,
			'TPTN' if use_TP_TN else 'label',
			str(n_per_side),
			'all' if all_splits else 'test',
			cluster_method,
			cluster_metric,
			str(dbscan_params['eps'] if cluster_method == 'dbscan' 
					else hdbscan_params['min_cluster_size']),
			str(dbscan_params['min_samples'] if cluster_method == 'dbscan' 
					else hdbscan_params['min_samples']),
			str(n_particles),
			str(int(datetime.now().timestamp()))
		])
		cluster_res_dir = os.path.join('cbseq_2_plots', cluster_res_dir)
		if not os.path.exists(cluster_res_dir):
			os.makedirs(cluster_res_dir)

	for start_pattern in starts_to_cluster:
		print(start_pattern)

		# Subset data
		pos_subset_data = subset_attr_data(
			pos_class_data,
			pos_STR_starts == start_pattern
		)
		neg_subset_data = subset_attr_data(
			neg_class_data,
			neg_STR_starts == start_pattern
		)

		# Get sequences to make features
		pos_subset_strings = np.array([
			np.array(list(c[-n_per_side-str_pad_size:-str_pad_size])) 
				for c in pos_subset_data['pre_strings']
		])
		neg_subset_strings = np.array([
			np.array(list(c[-n_per_side-str_pad_size:-str_pad_size])) 
				for c in neg_subset_data['pre_strings']
		])

		# Format data for distance metric
		if cluster_metric == 'hamming':
			X_pos = pos_subset_strings.copy()
			X_pos[X_pos == 'A'] = 0
			X_pos[X_pos == 'C'] = 1
			X_pos[X_pos == 'G'] = 2
			X_pos[X_pos == 'T'] = 3
			base_mapping = np.sort(np.unique(X_pos))
			X_pos = np.searchsorted(base_mapping, X_pos)

			X_neg = neg_subset_strings.copy()
			X_neg[X_neg == 'A'] = 0
			X_neg[X_neg == 'C'] = 1
			X_neg[X_neg == 'G'] = 2
			X_neg[X_neg == 'T'] = 3
			base_mapping = np.sort(np.unique(X_neg))
			X_neg = np.searchsorted(base_mapping, X_neg)
		else:
			X_pos = np.stack([
				(pos_subset_strings == 'A').astype(int),
				(pos_subset_strings == 'C').astype(int),
				(pos_subset_strings == 'G').astype(int),
				(pos_subset_strings == 'T').astype(int)
			]).transpose(1,0,2)
			X_pos = X_pos.reshape(X_pos.shape[0], -1)

			X_neg = np.stack([
				(neg_subset_strings == 'A').astype(int),
				(neg_subset_strings == 'C').astype(int),
				(neg_subset_strings == 'G').astype(int),
				(neg_subset_strings == 'T').astype(int)
			]).transpose(1,0,2)
			X_neg = X_neg.reshape(X_neg.shape[0], -1)


		# Cluster
		# Generate weights
		if use_weights:
			pos_attrs = pos_subset_data['pre'][
				:,:4:,-n_per_side-str_pad_size:-str_pad_size]
			neg_attrs = neg_subset_data['pre'][
				:,:4:,-n_per_side-str_pad_size:-str_pad_size]
			attr_mgm = np.max(
				np.abs(np.concatenate([pos_attrs, neg_attrs], axis=0)), 
				axis=1
			).mean(0)

		if use_weights and cluster_method == 'dbscan':
			print('Calculating distance matrices...')
			pos_dist_mat = metrics.pairwise_distances(
				X_pos,
				metric=cluster_metric,
				n_jobs=-1,
				w=attr_mgm / attr_mgm.sum()
			)
			neg_dist_mat = metrics.pairwise_distances(
				X_neg,
				metric=cluster_metric,
				n_jobs=-1,
				w=attr_mgm / attr_mgm.sum()
			)
			print("\tcomplete")
			pos_clust_labels = clusterer.fit_predict(pos_dist_mat)
			neg_clust_labels = clusterer.fit_predict(neg_dist_mat)
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
			for i in range(X_pos.shape[1]):
				expansion_inds.extend([i] * particle_dist[i])
			
			X_pos = X_pos[:, expansion_inds]
			X_neg = X_neg[:, expansion_inds]

			pos_clust_labels = clusterer.fit_predict(X_pos)
			neg_clust_labels = clusterer.fit_predict(X_neg)
		else:
			pos_clust_labels = clusterer.fit_predict(X_pos)
			neg_clust_labels = clusterer.fit_predict(X_neg)

		print("Positive class: {} clusters".format(len(np.unique(pos_clust_labels))))
		print(*zip(*np.unique(pos_clust_labels, return_counts=True)))

		print("Negative class: {} clusters".format(len(np.unique(neg_clust_labels))))
		print(*zip(*np.unique(neg_clust_labels, return_counts=True)))

		# Plot clusters for both classes
		pos_fig, _, pos_sorted_labels = plot_cluster_attn_and_ppm(
			pos_clust_labels, 
			pos_subset_data, 
			n_per_side,
			str_pad_size,
			label_desc='TP' if use_TP_TN else 'heterozygous',
			plt_height=.8
		)
		neg_fig, _, neg_sorted_labels = plot_cluster_attn_and_ppm(
			neg_clust_labels, 
			neg_subset_data, 
			n_per_side,
			str_pad_size,
			label_desc='TN' if use_TP_TN else 'non-heterozygous',
			plt_height=.8
		)

		# plot prediction confidence dists and attrs by cluster and class
		pos_conf_df = pd.DataFrame({
			'pred_conf': pos_subset_data['predictions'],
			'cluster': pos_clust_labels,
			'heterozygosity_score': pos_subset_data['sample_data'].label.values
		})
	
		pos_conf_attr_fig, axs = plt.subplots(1, 3, 
			figsize=(6.4, pos_fig.get_figheight())
		)
		sns.violinplot(
			x='pred_conf',
			y='cluster',
			orient='h',
			scale='count',
			order=pos_sorted_labels,
			cut=0,
			data=pos_conf_df,
			ax=axs[0]
		)
		sns.violinplot(
			x='heterozygosity_score',
			y='cluster',
			orient='h',
			order=pos_sorted_labels,
			cut=0,
			data=pos_conf_df,
			ax=axs[1]
		)
		sns.countplot(
			y='cluster',
			data=pos_conf_df,
			order=pos_sorted_labels,
			ax=axs[2]
		)
		# axs[1].set_xscale('log')
		axs[2].set_xlabel('Number of examples')
		axs[2].set_ylabel(None)

		abs_values = pos_conf_df['cluster'].value_counts(ascending=False)
		rel_values = pos_conf_df['cluster'].value_counts(ascending=False, normalize=True) * 100
		count_labels = [
			f'{abs_values[l]} ({rel_values[l]:.1f}%)' for l in pos_sorted_labels
		]
		axs[2].bar_label(container=axs[2].containers[0], labels=count_labels)

		# Negative equivalent plot (plot 4/4)
		neg_conf_df = pd.DataFrame({
			'pred_conf': neg_subset_data['predictions'],
			'cluster': neg_clust_labels,
			'heterozygosity_score': neg_subset_data['sample_data'].label.values
		})
	
		neg_conf_attr_fig, axs = plt.subplots(1, 3, 
			figsize=(6.4, neg_fig.get_figheight())
		)
		sns.violinplot(
			x='pred_conf',
			y='cluster',
			orient='h',
			scale='count',
			order=neg_sorted_labels,
			cut=0,
			data=neg_conf_df,
			ax=axs[0]
		)
		sns.violinplot(
			x='heterozygosity_score',
			y='cluster',
			orient='h',
			order=neg_sorted_labels,
			cut=0,
			data=neg_conf_df,
			ax=axs[1]
		)
		sns.countplot(
			y='cluster',
			data=neg_conf_df,
			order=neg_sorted_labels,
			ax=axs[2]
		)
		# axs[1].set_xscale('log')
		axs[2].set_xlabel('Number of examples')
		axs[2].set_ylabel(None)

		abs_values = neg_conf_df['cluster'].value_counts(ascending=False)
		rel_values = neg_conf_df['cluster'].value_counts(ascending=False, normalize=True) * 100
		count_labels = [
			f'{abs_values[l]} ({rel_values[l]:.1f}%)' for l in neg_sorted_labels
		]
		axs[2].bar_label(container=axs[2].containers[0], labels=count_labels)

		if save_plots:
			pos_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_pos_by_base.png'.format(start_pattern)
			))
			neg_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_neg_by_base.png'.format(start_pattern)
			))
			pos_conf_attr_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_pos_by_cluster.png'.format(start_pattern)
			))
			neg_conf_attr_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_neg_by_cluster.png'.format(start_pattern)
			))
		if show_plots:
			plt.show()
		
		plt.close()

