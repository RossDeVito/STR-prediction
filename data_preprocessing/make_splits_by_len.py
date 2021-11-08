"""Generates random train/val/test splits for samples based on a
sample_data.json file resulting from preprocess_STRs.py.
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns


# def make_random_splits(df, sample_data, train_ratio, test_ratio, rand_seed=None):
# 	train, val, test = np.split(
# 		df.sample(frac=1, random_state=rand_seed),	# shuffle	
# 		[int(train_ratio*len(df)), int((1-test_ratio)*len(df))]
# 	)

# 	# Make split json file
# 	splits_data = {'train': [], 'val': [], 'test': []}

# 	for i in train.idx.values.astype(int):
# 		splits_data['train'].append(sample_data[i])
# 	for i in val.idx.values.astype(int):
# 		splits_data['val'].append(sample_data[i])
# 	for i in test.idx.values.astype(int):
# 		splits_data['test'].append(sample_data[i])

# 	return splits_data

def make_rand_splits_by_STR_name(df, sample_data, train_ratio, test_ratio, rand_seed=None):
	"""Makes random train, val, and test splits with complements of each
	STR always in same split.
	"""
	other_inds, test_inds = next(GroupShuffleSplit(
		test_size=test_ratio, 
		n_splits=1, 
		random_state=rand_seed).split(df, groups=df['HipSTR_name']))
	train_inds, val_inds = next(GroupShuffleSplit(
		test_size=(1-test_ratio-train_ratio)/(1-test_ratio),
		n_splits=1,
		random_state=rand_seed).split(df.iloc[other_inds], groups=df.iloc[other_inds]['HipSTR_name']))

	# Make split json file
	splits_data = {'train': [], 'val': [], 'test': []}

	for i in train_inds:
		splits_data['train'].append(sample_data[i])
	for i in val_inds:
		splits_data['val'].append(sample_data[i])
	for i in test_inds:
		splits_data['test'].append(sample_data[i])

	return splits_data


if __name__ == '__main__':
	train_ratio = .7
	test_ratio = .15

	allowed_num_copies = {6}

	sample_data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	sample_data_path = os.path.join(sample_data_dir, 'sample_data.json')

	lab_samps_path = os.path.join('..', 'data', 'heterozygosity', 'labeled_samples_het.json')

	# Load sample data
	with open(sample_data_path) as fp:    
		sample_data = json.load(fp)
	with open(lab_samps_path) as fp:    
		labeled_samples = json.load(fp)

	# Get STR length info
	str_to_n_copies = dict()


	for samp in labeled_samples:
		str_to_n_copies[samp['HipSTR_name']] = int(samp['num_copies']) # truncate .5s

	# also truncate .5s
	STR_lengths = np.array(list(str_to_n_copies.values()))

	# # plot distiruibution of STR lengths
	# unique, counts = np.unique(STR_lengths, return_counts=True)

	# sns.displot(STR_lengths)
	# plt.show()

	# # Other way to do this
	min_data = []

	for samp in labeled_samples:
		min_data.append({
			'HipSTR_name': samp['HipSTR_name'],
			'num_copies': int(samp['num_copies']),
			'label': samp['label']
		})

	min_df = pd.DataFrame(min_data)
	min_df['is_het'] = min_df['label'] > 0
	counts_df = min_df.groupby(['num_copies',  'is_het']).count()
	ax = sns.barplot(x='num_copies', y='label', hue='is_het', data=counts_df.reset_index())
	ax.set_ylabel('number of samples')
	plt.show()

	
	
	# Set up for stratefied spliting for only specifed repeat counts
	idx = []
	labels = []
	str_name = []

	for i in range(len(sample_data)):
		if str_to_n_copies[sample_data[i]['HipSTR_name']] in allowed_num_copies:
			idx.append(i)
			labels.append(sample_data[i]['label'])
			str_name.append(sample_data[i]['HipSTR_name'])

	df = pd.DataFrame(
		np.vstack([idx, labels, str_name]).T, 
		columns=['idx', 'label', 'HipSTR_name']
	)

	other_inds, test_inds = next(GroupShuffleSplit(
		test_size=test_ratio, 
		n_splits=1, 
		random_state=36).split(df, groups=df['HipSTR_name']))
	train_inds, val_inds = next(GroupShuffleSplit(
		test_size=(1-test_ratio-train_ratio)/(1-test_ratio),
		n_splits=1,
		random_state=36).split(df.iloc[other_inds], groups=df.iloc[other_inds]['HipSTR_name']))

	# Generate splits
	split_1 = make_rand_splits_by_STR_name(df, sample_data, train_ratio, test_ratio, 147)
	split_2 = make_rand_splits_by_STR_name(df, sample_data, train_ratio, test_ratio, 36)
	split_3 = make_rand_splits_by_STR_name(df, sample_data, train_ratio, test_ratio, 23507)
	
	# Save splits as JSON
	n_copies_str = '_'.join(map(str, sorted(allowed_num_copies)))

	with open(os.path.join(sample_data_dir, 'split_1_nc{}.json'.format(n_copies_str)), 'w') as fp:
		json.dump(split_1, fp, indent=4)
	with open(os.path.join(sample_data_dir, 'split_2_nc{}.json'.format(n_copies_str)), 'w') as fp:
		json.dump(split_2, fp, indent=4)
	with open(os.path.join(sample_data_dir, 'split_3_nc{}.json'.format(n_copies_str)), 'w') as fp:
		json.dump(split_3, fp, indent=4)