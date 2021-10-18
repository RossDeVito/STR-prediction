"""Generates random train/val/test splits for samples based on a
sample_data.json file resulting from preprocess_STRs.py.
"""

import os
import json

import numpy as np
import pandas as pd


def make_random_splits(df, sample_data, train_ratio, test_ratio, rand_seed=None):
	train, val, test = np.split(
		df.sample(frac=1, random_state=rand_seed),	# shuffle	
		[int(train_ratio*len(df)), int((1-test_ratio)*len(df))]
	)

	# Make split json file
	splits_data = {'train': [], 'val': [], 'test': []}

	for i in train.idx.values.astype(int):
		splits_data['train'].append(sample_data[i])
	for i in val.idx.values.astype(int):
		splits_data['val'].append(sample_data[i])
	for i in test.idx.values.astype(int):
		splits_data['test'].append(sample_data[i])

	return splits_data


if __name__ == '__main__':
	train_ratio = .7
	test_ratio = .15

	sample_data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples')
	sample_data_path = os.path.join(sample_data_dir, 'sample_data.json')

	# Load sample data
	with open(sample_data_path) as fp:    
		sample_data = json.load(fp)
	
	# Set up for stratefied spliting
	idx = []
	labels = []

	for i in range(len(sample_data)):
		idx.append(i)
		labels.append(sample_data[i]['label'])

	df = pd.DataFrame(np.vstack([idx, labels]).T, columns=['idx', 'label'])

	# Generate splits
	split_1 = make_random_splits(df, sample_data, train_ratio, test_ratio, 147)
	split_2 = make_random_splits(df, sample_data, train_ratio, test_ratio, 36)
	split_3 = make_random_splits(df, sample_data, train_ratio, test_ratio, 23507)
	
	# Save splits as JSON
	with open(os.path.join(sample_data_dir, 'split_1.json'), 'w') as fp:
		json.dump(split_1, fp, indent=4)
	with open(os.path.join(sample_data_dir, 'split_2.json'), 'w') as fp:
		json.dump(split_2, fp, indent=4)
	with open(os.path.join(sample_data_dir, 'split_3.json'), 'w') as fp:
		json.dump(split_3, fp, indent=4)