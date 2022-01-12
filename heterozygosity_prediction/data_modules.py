""" Data modules for handle V2 data. Representations of data to be passes
	to models are now created JIT instead of ahead of time and stored to disk.
	This should result in much more flexibilty and probably won't be a 
	bottleneck.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class STRDataset(Dataset):
	""" Dataset for loading STR data and creating feature matrices. 

	When min_boundary_STR_pos != max_boundary_STR_pos, an int is randomly
	selected from the range of min_boundary_STR_pos to max_boundary_STR_pos
	inclusive each time __getitem__ is called.

	Min and max copy number are used to reduce dataset to just the desired
	copy number range. Values are inclusive. If None, then no filtering is
	done for that criteria. Because the splits are stratified by copy number,
	everything should still be balanced when this is done.
	
	Attributes:
		data_file (str): path to json file containing data
		split_name (str): name of split column to use
		split_type (int): 0: 'train', 1: 'val', or 2: 'test' corresponding
			to values in split_name column of data_file. If None, all
		incl_STR_feat (bool): whether to include 5th row of feature matrix
			with binary label if position is part of the central STR
		min_boundary_STR_pos (int): minimum number of STRs loci to include
		max_boundary_STR_pos (int): maximum number of STRs loci to include
		window_size (int): size of window to include around STR
		min_copy_num (int): minimum copy number to include STR in Dataset
		max_copy_num (int): maximum copy number to include STR in Dataset
		bp_dist_units (float): distance feature will be in units of 
			(# of bp) / bp_dist_base
		as_tensors (bool): whether to return tensors instead of numpy arrays
			for feature matrices
		return_data (bool): whether to return the data object from the
			master sheet in addition to the feature matrices, labels, and
			seqs as strings
		return_strings (bool): whether to return the sequences as strings
	"""
	def __init__(self, data_file, split_name=None, split_type=None,
			incl_STR_feat=True, min_boundary_STR_pos=2, max_boundary_STR_pos=2,
			window_size=128, min_copy_num=None, max_copy_num=None,
			bp_dist_units=1000.0, as_tensors=True, 
			return_data=False, return_strings=False):
		"""Inits STRDataset with data_file and desired split or all 
		data (default).
		"""
		self.data_file = data_file
		self.split_name = split_name
		self.split_type = split_type
		self.incl_STR_feat = incl_STR_feat
		self.min_boundary_STR_pos = min_boundary_STR_pos
		self.max_boundary_STR_pos = max_boundary_STR_pos
		self.window_size = window_size
		self.min_copy_num = min_copy_num
		self.max_copy_num = max_copy_num
		self.bp_dist_units = bp_dist_units
		self.as_tensors = as_tensors
		self.return_data = return_data
		self.return_strings = return_strings

		# Load data as DataFrame
		self.data = pd.read_json(data_file)

		# Reduce data to just desired split
		if split_name is not None and split_type is not None:
			self.data = self.data[self.data[split_name] == split_type]

		# Reduce data to just desired copy number range
		if min_copy_num is not None:
			self.data = self.data[self.data['num_copies'] >= min_copy_num]
		if max_copy_num is not None:
			self.data = self.data[self.data['num_copies'] <= max_copy_num]

		# Reset indices to just required size. Still have HipSTR names as
		#  unique global keys if needed.
		self.data = self.data.reset_index(drop=True)

	def __len__(self):
		"""Returns length of dataset."""
		return len(self.data)

	def num_feat_channels(self):
		"""Returns number of feature channels in matrix returned by 
		__getitem__.
		"""
		if self.incl_STR_feat:
			return 6
		else:
			return 5

	def make_feature_matrix(self, seq_string, num_STR_loci, pre_STR):
		"""Makes tensor of features for given seqence."""
		seq = np.array(list(seq_string))
		unk = (seq == 'N').astype(int) / 4

		A = unk + (seq == 'A')
		C = unk + (seq == 'C')
		G = unk + (seq == 'G')
		T = unk + (seq == 'T')

		if pre_STR:
			dists = np.array(
				list(range(-len(seq_string) + num_STR_loci, num_STR_loci, 1))
			) / self.bp_dist_units
			dists[-num_STR_loci:] = 0
		else:
			dists = np.array(
				list(range(-num_STR_loci+1, len(seq_string) - num_STR_loci + 1, 1))
			) / self.bp_dist_units
			dists[:num_STR_loci] = 0

		if self.incl_STR_feat:
			if pre_STR:
				is_STR = np.zeros(len(seq_string))
				is_STR[-num_STR_loci:] = 1
			else:
				is_STR = np.zeros(len(seq_string))
				is_STR[:num_STR_loci] = 1
			feat_mat = np.vstack((A, C, G, T, dists, is_STR))
		else:
			feat_mat = np.vstack((A, C, G, T, dists))

		if self.as_tensors:
			return torch.tensor(feat_mat)
		else:
			return feat_mat

	def make_features(self, STR_data):
		# Check if number of added STR loci is single value or range
		if self.min_boundary_STR_pos == self.max_boundary_STR_pos:
			num_to_add = self.min_boundary_STR_pos
		else:
			num_to_add = np.random.randint(self.min_boundary_STR_pos,
											self.max_boundary_STR_pos + 1)

		# Create stings for pre and post STR seqs
		pre_STR_seq = (STR_data.pre_seq[-(self.window_size - num_to_add):] 
						+ STR_data.str_seq[:num_to_add])

		post_STR_seq = (STR_data.str_seq[-num_to_add:]
						+ STR_data.post_seq[:(self.window_size - num_to_add)])

		if self.return_strings:
			return {
				'pre_STR_seq': pre_STR_seq,
				'pre_STR_feats': self.make_feature_matrix(pre_STR_seq, num_to_add, True),
				'post_STR_seq': post_STR_seq,
				'post_STR_feats': self.make_feature_matrix(post_STR_seq, num_to_add, False)
			}
		else:
			return {
				'pre_STR_feats': self.make_feature_matrix(pre_STR_seq, num_to_add, True),
				'post_STR_feats': self.make_feature_matrix(post_STR_seq, num_to_add, False)
			}

	def __getitem__(self, idx):
		"""Returns a single sample from the dataset.

		Args:
			idx (int): index of sample to return

		Returns:
			dict with keys:
				'data': all data in row of data_file for sample as object
				'feat_mat': feature matrix for sample as tensor
				'label': binary label for sample as int
		"""
		STR_data = self.data.iloc[idx]
		pre_post_feats = self.make_features(STR_data)

		if self.return_data:
			return {
				'data': STR_data,
				'label': STR_data.binary_label,
				**pre_post_feats
			}
		else:
			return {
				'label': STR_data.binary_label,
				**pre_post_feats
			}


def STR_data_collate_fn(batch):
	"""Collate function for PyTorch DataLoader.

	Args:
		batch (list): list of dicts returned by __getitem__

	Returns:
		dict with keys:
			'data': list of all data in row of data_file for sample as object
			'feat_mat': list of feature matrices for samples as tensor
			'label': list of binary labels for samples as int
	"""
	if 'data' in batch[0].keys() and 'pre_STR_seq' in batch[0].keys():
		return {
			'data': [sample['data'] for sample in batch],
			'pre_STR_seq': [sample['pre_STR_seq'] for sample in batch],
			'post_STR_seq': [sample['post_STR_seq'] for sample in batch],
			'pre_STR_feats': torch.stack(
				[sample['pre_STR_feats'] for sample in batch]).float(),
			'post_STR_feats': torch.stack(
				[sample['post_STR_feats'] for sample in batch]).float(),
			'label': torch.tensor([sample['label'] for sample in batch]).long()
		}
	elif 'data' in batch[0].keys():
		return {
			'data': [sample['data'] for sample in batch],
			'pre_STR_feats': torch.stack(
				[sample['pre_STR_feats'] for sample in batch]).float(),
			'post_STR_feats': torch.stack(
				[sample['post_STR_feats'] for sample in batch]).float(),
			'label': torch.tensor([sample['label'] for sample in batch]).long()
		}
	elif 'pre_STR_seq' in batch[0].keys():
		return {
			'pre_STR_seq': [sample['pre_STR_seq'] for sample in batch],
			'post_STR_seq': [sample['post_STR_seq'] for sample in batch],
			'pre_STR_feats': torch.stack(
				[sample['pre_STR_feats'] for sample in batch]).float(),
			'post_STR_feats': torch.stack(
				[sample['post_STR_feats'] for sample in batch]).float(),
			'label': torch.tensor([sample['label'] for sample in batch]).long()
		}
	else:
		return {
			'pre_STR_feats': torch.stack(
				[sample['pre_STR_feats'] for sample in batch]).float(),
			'post_STR_feats': torch.stack(
				[sample['post_STR_feats'] for sample in batch]).float(),
			'label': torch.tensor([sample['label'] for sample in batch]).long()
		}


class STRDataModule(pl.LightningDataModule):
	"""Lightning style data module for STR heterozygosity pred.
	
	Attributes:
		see STRDataset
		batch_size (int): batch size for data loader
		num_workers (int): number of workers for data loader
		shuffle (bool): whether to shuffle data in data loader
	"""
	def __init__(self, data_file, split_name, 
			batch_size=32, num_workers=0, shuffle=True,
			incl_STR_feat=True, min_boundary_STR_pos=2, max_boundary_STR_pos=2,
			window_size=128, min_copy_num=None, max_copy_num=None,
			bp_dist_units=1000.0, return_data=False, return_strings=False):
		super().__init__()
		self.data_file = data_file
		self.split_name = split_name
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.shuffle = shuffle
		self.incl_STR_feat = incl_STR_feat
		self.min_boundary_STR_pos = min_boundary_STR_pos
		self.max_boundary_STR_pos = max_boundary_STR_pos
		self.window_size = window_size
		self.min_copy_num = min_copy_num
		self.max_copy_num = max_copy_num
		self.bp_dist_units = bp_dist_units
		self.return_data = return_data
		self.return_strings = return_strings

	def setup(self, stage=None):
		self.train_data = STRDataset(self.data_file, self.split_name, 0,
			self.incl_STR_feat, self.min_boundary_STR_pos, 
			self.max_boundary_STR_pos, self.window_size, self.min_copy_num, 
			self.max_copy_num, self.bp_dist_units, 
			return_data=self.return_data,
			return_strings=self.return_strings
		)
		self.val_data = STRDataset(self.data_file, self.split_name, 1,
			self.incl_STR_feat, self.min_boundary_STR_pos, 
			self.max_boundary_STR_pos, self.window_size, self.min_copy_num, 
			self.max_copy_num, self.bp_dist_units,
			return_data=self.return_data,
			return_strings=self.return_strings
		)
		self.test_data = STRDataset(self.data_file, self.split_name, 2,
			self.incl_STR_feat, self.min_boundary_STR_pos, 
			self.max_boundary_STR_pos, self.window_size, self.min_copy_num, 
			self.max_copy_num, self.bp_dist_units,
			return_data=self.return_data,
			return_strings=self.return_strings
		)

	def train_dataloader(self):
		return DataLoader(self.train_data, batch_size=self.batch_size,
			shuffle=self.shuffle, num_workers=self.num_workers,
			collate_fn=STR_data_collate_fn)

	def val_dataloader(self):
		return DataLoader(self.val_data, batch_size=self.batch_size,
			num_workers=self.num_workers, collate_fn=STR_data_collate_fn)

	def test_dataloader(self):
		return DataLoader(self.test_data, batch_size=self.batch_size,
			num_workers=self.num_workers, collate_fn=STR_data_collate_fn)

	def num_feat_channels(self):
		"""Returns number of feature channels in matrix returned by 
		__getitem__.
		"""
		if self.incl_STR_feat:
			return 6
		else:
			return 5


if __name__ == '__main__':
	__spec__ = None
	torch.set_printoptions(edgeitems=8, linewidth=400)

	# Testing
	data_path = os.path.join('..', 'data', 'heterozygosity', 
								'sample_data_V2_repeat_var.json')

	# ds_all = STRDataset(data_path)
	# ds_0 = STRDataset(data_path, 'split_1', split_type=0)
	ds_1 = STRDataset(data_path, 'split_1', split_type=0, max_copy_num=15)

	data_mod = STRDataModule(data_path, 'split_1', num_workers=1, 
		min_boundary_STR_pos=4,
		max_boundary_STR_pos=6,
		max_copy_num=15,
	)
	data_mod.setup()
	train_dataloader = data_mod.train_dataloader()
	batch1 = next(iter(train_dataloader))

	# data_mod = STRDataModule(data_path, 'split_1', num_workers=3, return_strings=True)
	# data_mod.setup()
	# train_dataloader = data_mod.train_dataloader()
	# batch2 = next(iter(train_dataloader))

	# data_mod = STRDataModule(data_path, 'split_1', num_workers=3, return_data=True)
	# data_mod.setup()
	# train_dataloader = data_mod.train_dataloader()
	# batch3 = next(iter(train_dataloader))

	# data_mod = STRDataModule(data_path, 'split_1', num_workers=3, 
	# 	return_data=True, return_strings=True)
	# data_mod.setup()
	# train_dataloader = data_mod.train_dataloader()
	# batch4 = next(iter(train_dataloader))

	# data_mod = STRDataModule(data_path, 'split_1', num_workers=3)
	# data_mod.setup()
	# train_dataloader = data_mod.train_dataloader()
	# batch5 = next(iter(train_dataloader))