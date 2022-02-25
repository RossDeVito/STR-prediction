import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(pre_mat, post_mat, pre_str, post_str, 
		desc=None, label=None, pred=None, cmap='PRGn'):
	max_attr = max(pre_mat.max(), post_mat.max())
	min_attr = min(pre_mat.min(), post_mat.min())

	fig, ax = plt.subplots(1, 2, figsize=(35, 3))
	sns.heatmap(
		data=pre_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance', 'is_STR'],
		xticklabels=pre_str,
		ax=ax[0],
		cmap=cmap
	)
	sns.heatmap(
		data=post_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance', 'is_STR'],
		xticklabels=post_str,
		ax=ax[1],
		cmap=cmap
	)
	plt.tight_layout()

	if desc is not None:
		if label is not None and pred is not None:
			plt.suptitle('{}    label: {}, pred: {}'.format(desc, label, pred))
		else:
			plt.suptitle('{}'.format(desc))
	elif label is not None and pred is not None:
		plt.suptitle('label: {}, pred: {}'.format(label, pred))

	return fig, ax


def plot_heatmap_in_axs(pre_mat, post_mat, axs, 
		pre_str=None, post_str=None, desc=None, label=None, 
		pred=None, cmap='PRGn'):
	max_attr = max(pre_mat.max(), post_mat.max())
	min_attr = min(pre_mat.min(), post_mat.min())

	sns.heatmap(
		data=pre_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance', 'is_STR'],
		xticklabels=[],
		ax=axs[0],
		cmap=cmap,
		cbar=False
	)
	sns.heatmap(
		data=post_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance', 'is_STR'],
		xticklabels=[],
		ax=axs[1],
		cmap=cmap,
		cbar_ax=axs[2]
	)
	plt.tight_layout()

	if desc is not None:
		if label is not None and pred is not None:
			axs[0].set_title('{}    label: {}, pred: {}'.format(desc, label, pred))
		else:
			axs[0].set_title('{}'.format(desc))
	elif label is not None and pred is not None:
		axs[0].set_title('label: {}, pred: {}'.format(label, pred))

	return axs


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
	save_plots = True
	show_plots = False
	str_motif_len = 2
	str_pad_size = 6

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
		attr_data, (attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
	)
	tn_data = subset_attr_data(
		attr_data, (attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
	)

	# Create plot save dir
	if save_plots:
		plot_save_dir = os.path.join('plots', motif_type, model_dir)
		if not os.path.exists(plot_save_dir):
			os.makedirs(plot_save_dir)

	# plot top 5 TPs for each motif
	top_n = 50

	for motif in ['AG']:#tp_data['sample_data'].alpha_motif.unique():
		subset_mask = tp_data['sample_data'].alpha_motif == motif
		subset_data = subset_attr_data(tp_data, subset_mask)
		tp_inds = np.argsort(-subset_data['predictions'])[:top_n]
		fig, axs = plt.subplots(
			top_n, 
			3, 
			figsize=(20, .9*top_n),
			gridspec_kw={'width_ratios': [1, 1, .03]}
		)

		for plot_ind,ind in enumerate(tp_inds):
			plot_heatmap_in_axs(
				subset_data['pre'][ind],
				subset_data['post'][ind],
				axs[plot_ind],
				ind
			)
		fig.suptitle(
			'Top {} TPs for motif {}'.format(top_n, motif),
			y=.9995,
			size='xx-large'
		)
		plt.tight_layout()
	
		if save_plots:
			plt.savefig(
				os.path.join(
					plot_save_dir, 
					'top_{}_TPs_{}_{}.png'.format(top_n, motif, attr_file.split('.')[0])
				)
			)
		if show_plots:
			plt.show()
		plt.close()

	# # plot top n TNs for each motif
	# for motif in tn_data['sample_data'].alpha_motif.unique():
	# 	subset_mask = tn_data['sample_data'].alpha_motif == motif
	# 	subset_data = subset_attr_data(tn_data, subset_mask)
	# 	tn_inds = np.argsort(subset_data['predictions'])[:top_n]
	# 	fig, axs = plt.subplots(
	# 		top_n, 
	# 		3, 
	# 		figsize=(20, .9*top_n),
	# 		gridspec_kw={'width_ratios': [1, 1, .03]}
	# 	)

	# 	for plot_ind,ind in enumerate(tn_inds):
	# 		plot_heatmap_in_axs(
	# 			subset_data['pre'][ind],
	# 			subset_data['post'][ind],
	# 			axs[plot_ind],
	# 			ind
	# 		)
	# 	fig.suptitle(
	# 		'Top {} TNs for motif {}'.format(top_n, motif),
	# 		y=.9995,
	# 		size='xx-large')
	# 	plt.tight_layout()
	
	# 	if save_plots:
	# 		plt.savefig(
	# 			os.path.join(
	# 				plot_save_dir, 
	# 				'top_{}_TNs_{}_{}.png'.format(top_n, motif, attr_file.split('.')[0])
	# 			)
	# 		)
	# 	if show_plots:
	# 		plt.show()
	# 	plt.close()