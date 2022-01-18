"""Creates input feature representation for labeled samples, finds
corresponding feature string and chromosome range, removes STRs with
length over threshold, and saves resulting samples in a JSON fine
"""

import json
import os
import sys
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def make_feat_mat(seq_string, distance):
	bases = np.array(list(seq_string))
	unk = (bases == 'N').astype(int) / 4

	A = unk + (bases == 'A')
	C = unk + (bases == 'C')
	G = unk + (bases == 'G')
	T = unk + (bases == 'T')

	return np.vstack((A, C, G, T, distance))


def is_perfect_repeat(seq, motif):
	# Check is motif matches forward
	if seq[:len(motif)] == motif:
		pass
	# If matches reversed, reverse motif
	elif seq[:len(motif)] == motif[::-1]:
		motif = motif[::-1]
	else:
		return False

	# Check that seq is just the motif repeated
	for i in range(len(seq)):
		if seq[i] != motif[i%len(motif)]:
			return False

	return True


def make_stratified_splits(group_label, train_ratio, test_ratio, rand_seed=None):
	""" Makes random train, val, and test splits stratified by some 
		group_label. Returns results as a vector where 0 is train,
		1 is val, and 2 is test.
	"""
	other_inds, test_inds = next(StratifiedShuffleSplit(
		test_size=test_ratio, 
		n_splits=1, 
		random_state=rand_seed).split(group_label, group_label))
	train_inds, val_inds = next(StratifiedShuffleSplit(
		test_size=(1-test_ratio-train_ratio)/(1-test_ratio),
		n_splits=1,
		random_state=rand_seed).split(group_label[other_inds], group_label[other_inds]))

	# Make return vector of split labels
	split_labels = np.zeros(len(group_label), dtype=int)
	split_labels[other_inds[val_inds]] = 1
	split_labels[test_inds] = 2

	return split_labels

if __name__ == '__main__':
	# Parameters
	max_STR_len = 50
	output_seq_len = 1000

	min_num_called = 100 # if None will skip, for het task

	# Motifs not in this list will be removed. Remember labeled_samples
	#  already includes complements
	motif_types = ['T']

	remove_imperfect_repeats = True # Remove STRs that are not perfect repeats of motif

	# Load labeled STRs to be preprocessed
	samp_dir = os.path.join('..', 'data', 'heterozygosity')
	samp_fname = 'labeled_samples_T_het.json'
	samp_path = os.path.join(samp_dir, samp_fname)

	with open(samp_path) as fp:    
		samples = json.load(fp)

	filtered_sample_data = []
	labels = []

	# Filter samples by STR length, then create formatted output_seq_len samples
	for i in tqdm(range(len(samples)), file=sys.stdout):
		samp_dict = samples.pop(0)

		# filter out by STR motif type
		if samp_dict['motif'] not in motif_types:
			continue

		# filter out by imperfect repeats
		if remove_imperfect_repeats and (
				not is_perfect_repeat(samp_dict['str_seq'], samp_dict['motif'])):
			continue

		# filter out by STR length
		if samp_dict['motif_len'] * samp_dict['num_copies'] > max_STR_len:
			continue

		# filter out by min num called
		if min_num_called is not None and samp_dict['num_called'] < min_num_called:
			continue

		samp_dict['binary_label'] = int(samp_dict['label'] > 0)
		filtered_sample_data.append(samp_dict)

		# # for dev
		# if i > 10000:
		# 	break

	# Generate splits based on copy number, such that dataloader can restrict
	#  the copy number range and have balanced groups down the road
	samp_df = pd.DataFrame(filtered_sample_data)
	samp_df['split_1'] = make_stratified_splits(samp_df['num_copies'], 0.7, 0.15, 36)
	samp_df['split_2'] = make_stratified_splits(samp_df['num_copies'], 0.7, 0.15, 147)
	samp_df['split_3'] = make_stratified_splits(samp_df['num_copies'], 0.7, 0.15, 12151997)

	# Plot copy number distribution by split to verify that splits are balanced
	# sns.countplot(x='num_copies', hue='split_1', data=samp_df)
	# plt.show()
	# sns.catplot(x='num_copies', hue='split_1', row='binary_label', data=samp_df,
	# 	kind='count', aspect=2)
	# plt.show()
	# sns.catplot(x='num_copies', hue='split_2', row='binary_label', data=samp_df,
	# 	kind='count', aspect=2)
	# plt.show()
	# sns.catplot(x='num_copies', hue='split_3', row='binary_label', data=samp_df,
	# 	kind='count', aspect=2)
	# plt.show()

	# Save JSON of preprocessed samples
	save_dir = os.path.join('..', 'data', 'heterozygosity',)
	this_sample_set_fname = 'sample_data_T_V2_repeat_var.json'
	save_path = os.path.join(save_dir, this_sample_set_fname)

	samp_df.to_json(save_path, orient='records', indent=4)