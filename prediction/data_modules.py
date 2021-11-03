import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class STRDataset(Dataset):
	def __init__(self, data_dir, split_file, split_name):
		self.data_dir = data_dir
		self.split_file = split_file
		self.split_name = split_name

		# load data for samples in split
		with open(os.path.join(self.data_dir, self.split_file)) as fp:    
			self.split_data = json.load(fp)[self.split_name]

	def __len__(self):
		return len(self.split_data)

	def __getitem__(self, idx):
		"""Return sequence in matrix form, sequence as string, chromosome
		location, and label."""
		item_data = self.split_data[idx]
		feat_mat = np.load(os.path.join(self.data_dir, item_data['fname']))

		return {
			'feat_mat': torch.tensor(feat_mat).float(),
			'seq_string': item_data['seq_string'],
			'chr_loc': item_data['chr_loc'],
			'label': torch.tensor(item_data['label']).float()
		}


class STRDatasetNonzeroClass(STRDataset):
	def __init__(self, data_dir, split_file, split_name):
		super().__init__(data_dir, split_file, split_name)

	def __getitem__(self, idx):
		"""Return sequence in matrix form, sequence as string, chromosome
		location, and label."""
		item_data = self.split_data[idx]
		feat_mat = np.load(os.path.join(self.data_dir, item_data['fname']))

		return {
			'feat_mat': torch.tensor(feat_mat).float(),
			'seq_string': item_data['seq_string'],
			'chr_loc': item_data['chr_loc'],
			'label': torch.tensor(item_data['label'] > 0).float(),
			'label_val': torch.tensor(item_data['label']).float()
		}


class STRDataModule(pl.LightningDataModule):
	def __init__(self, data_dir, split_file, batch_size = 32, num_workers = 1,
					shuffle = True):
		super().__init__()
		self.data_dir = data_dir
		self.split_file = split_file
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.shuffle = shuffle

	def setup(self, stage=None):
		self.train_data = STRDataset(self.data_dir, self.split_file, 'train')
		self.val_data = STRDataset(self.data_dir, self.split_file, 'val')
		self.test_data = STRDataset(self.data_dir, self.split_file, 'test')

	def train_dataloader(self):
		return DataLoader(self.train_data, batch_size=self.batch_size,
			num_workers=self.num_workers, shuffle=self.shuffle)

	def val_dataloader(self):
		return DataLoader(self.val_data, batch_size=self.batch_size,
			num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.test_data, batch_size=self.batch_size,
			num_workers=self.num_workers)


class STRDataModuleZonzeroClass(STRDataModule):
	"""Takes labels from regression and makes into >0 classification problem."""
	def __init__(self, data_dir, split_file, batch_size = 32, num_workers = 1,
					shuffle = True):
		super().__init__(data_dir, split_file, batch_size, num_workers, shuffle)

	def setup(self, stage=None):
		self.train_data = STRDatasetNonzeroClass(self.data_dir, self.split_file, 'train')
		self.val_data = STRDatasetNonzeroClass(self.data_dir, self.split_file, 'val')
		self.test_data = STRDatasetNonzeroClass(self.data_dir, self.split_file, 'test')


""" Task specific data modules """

# STR heterozygosity from pre- and post-sequeneces
class STRHetPrePostDataset(STRDataset):
	def __init__(self, data_dir, split_file, split_name, n_per_side=500):
		super().__init__(data_dir, split_file, split_name)
		self.n_per_side = n_per_side

	def __getitem__(self, idx):
		item_data = self.split_data[idx]
		feat_mat = np.load(os.path.join(self.data_dir, item_data['fname']))

		return {
			'pre_feat_mat': torch.tensor(feat_mat[:, :self.n_per_side]).float(),
			'post_feat_mat': torch.tensor(feat_mat[:, -self.n_per_side:]).float(),
			'pre_seq_string': item_data['seq_string'][:self.n_per_side],
			'post_seq_string': item_data['seq_string'][-self.n_per_side:],
			'chr_loc': item_data['chr_loc'],
			'label': torch.tensor(item_data['label'] > 0).float(),
			'label_val': torch.tensor(item_data['label']).float()
		}


class STRHetPrePostDataModule(STRDataModule):
	"""Takes labels from regression and makes into >0 classification problem."""
	def __init__(self, data_dir, split_file, batch_size = 32, num_workers = 1,
					shuffle = True, n_per_side = 500):
		super().__init__(data_dir, split_file, batch_size, num_workers, shuffle)
		self.n_per_side = n_per_side

	def setup(self, stage=None):
		self.train_data = STRHetPrePostDataset(
			self.data_dir, self.split_file, 'train', self.n_per_side)
		self.val_data = STRHetPrePostDataset(
			self.data_dir, self.split_file, 'val', self.n_per_side)
		self.test_data = STRHetPrePostDataset(
			self.data_dir, self.split_file, 'test', self.n_per_side)


if __name__ == "__main__":
	# Testing
	# data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples2')
	# split_file = 'split_1.json'

	# STR_data = STRDataModule(data_dir, split_file, num_workers=2)
	# STR_data.setup()

	# print(next(iter(STR_data.val_dataloader()))['feat_mat'].shape)

	# STR_data_nz = STRDataModuleZonzeroClass(data_dir, split_file, num_workers=2)
	# STR_data_nz.setup()

	# print(next(iter(STR_data_nz.val_dataloader()))['feat_mat'].shape)

	# STR het prepost
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	split_file = 'split_1.json'

	STR_data = STRHetPrePostDataModule(data_dir, split_file, num_workers=2, 
										n_per_side=500)
	STR_data.setup()
	print(next(iter(STR_data.val_dataloader()))['pre_feat_mat'].shape)
	print(next(iter(STR_data.val_dataloader()))['post_feat_mat'].shape)
	print(STR_data.val_data[0])