import os
import copy

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


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

	tp_data = subset_attr_data(
		attr_data, (attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
	)
	tn_data = subset_attr_data(
		attr_data, (attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
	)

	# Find best window for pre
	pre_seq = True # to use negative indices
	find_max = True
	window_size = 7
	str_motif_len = 2
	str_pad_size = 4 # set to less that real padding or 0 to include str padding

	if pre_seq:
		attrs = tp_data['pre']
		seqs = tp_data['pre_strings']
	else:
		attrs = tp_data['post']
		seqs = tp_data['post_strings']

	# subset attributions to base features
	attrs = attrs[:, :4]
	
	# get best window for each sequence
	best_windows = []

	for a,s in tqdm(zip(attrs, seqs), total=len(seqs)):
		if pre_seq:
			# NOTE: saved indices are inclusive
			i_start = -a.shape[1]
			i_end = i_start + window_size
			str_boundary = s[-str_pad_size : -str_pad_size+str_motif_len]
		else:
			i_start = 0 + str_pad_size
			i_end = i_start + window_size
			str_boundary = s[str_pad_size-str_motif_len : str_pad_size]

		# Score windows
		window_vals = []
		window_inds = []
		while (pre_seq and i_end <= (0 - str_pad_size)) or \
					(not pre_seq and i_end <= len(s)):
			window = a[:, i_start:i_end or None]
			assert window.shape[1] == window_size

			if find_max:
				window_vals.append(window.max(axis=0).sum())
			else:
				window_vals.append(window.min(axis=0).sum())

			window_inds.append((i_start, i_end))
			i_start += 1
			i_end += 1

		# Find and save best window
		if find_max:
			best_ind = np.argmax(window_vals)
		else:
			best_ind = np.argmin(window_vals)

		best_windows.append({
			'score': window_vals[best_ind],
			'indices': window_inds[best_ind],
			'seq': s[window_inds[best_ind][0]:window_inds[best_ind][1] or None],
			'str_boundary': str_boundary
		})

	window_df = pd.DataFrame(best_windows)

	# Analysis
	print(window_df.seq.value_counts())

	# aggregate by seq
	agg_df = window_df.groupby('seq', as_index=False).agg({
		'score': 'mean',
		'indices': lambda x: list(x),
		'str_boundary': lambda x: list(x)
	})
	agg_df['count'] = agg_df.indices.apply(len)
	agg_df['unique_inds'] = agg_df.indices.apply(
		lambda x: dict(pd.Series(x).value_counts())
	)
	agg_df['str_boundary_counts'] = agg_df.str_boundary.apply(
		lambda x: dict(pd.Series(x).value_counts())
	)
	agg_df = agg_df.sort_values('count', ascending=not find_max)