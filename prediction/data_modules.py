import os
import json

import numpy as np
from numpy.lib import stride_tricks
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
	""" Dataset for STR heterozygosity from pre- and post-sequences. 

	To use with 5 feature (A, C, G, T, dist) matrix sequence 
	representations, use tokenizer = None.

	If using with BERT based feature extractor (DNABERT), use
	tokenizer = huggingface type pretrained tokenizer.

	Args:
		data_dir: directory containing data
		split_file: file containing data splits
		split_name: name of split to use
		n_per_side: number of samples to take from each side of sequence
		tokenizer: huggingface tokenizer to use for BERT, or None for
			more basic 5-feature matrix sequence representation
		k: Tokenizer will be used to split sequences into k-mers.
	"""	
	def __init__(self, data_dir, split_file, split_name, n_per_side=500,
					tokenizer=None, k=5, is_binary=True):
		super().__init__(data_dir, split_file, split_name)
		self.n_per_side = n_per_side
		self.tokenizer = tokenizer
		self.k = k
		self.is_binary = is_binary

	def _get_item_5_feat_mat(self, item_data):
		feat_mat = np.load(os.path.join(self.data_dir, item_data['fname']))

		if self.is_binary:
			return {
				'pre_feat_mat': torch.tensor(feat_mat[:, :self.n_per_side]).float(),
				'post_feat_mat': torch.tensor(feat_mat[:, -self.n_per_side:]).float(),
				'pre_seq_string': item_data['seq_string'][:self.n_per_side],
				'post_seq_string': item_data['seq_string'][-self.n_per_side:],
				'chr_loc': item_data['chr_loc'],
				'label': torch.tensor(item_data['label'] > 0).float(),
				'label_val': torch.tensor(item_data['label']).float()
			}
		else:
			return {
				'pre_feat_mat': torch.tensor(feat_mat[:, :self.n_per_side]).float(),
				'post_feat_mat': torch.tensor(feat_mat[:, -self.n_per_side:]).float(),
				'pre_seq_string': item_data['seq_string'][:self.n_per_side],
				'post_seq_string': item_data['seq_string'][-self.n_per_side:],
				'chr_loc': item_data['chr_loc'],
				'label': torch.tensor(item_data['label']).float()
			}

	def tokenize_seq_str(self, seq_str):
		""" Pre-tokenize sequences into kmers using sliding window,
			then tokenize each kmer.
		"""
		# Pre-tokenize sequences into kmers
		bases = np.fromiter(seq_str, (np.compat.unicode, 1))
		kmers = np.apply_along_axis(
			lambda row: row.astype('|S1').tostring().decode('utf-8'),
			axis=1,
			arr=stride_tricks.sliding_window_view(bases, window_shape=self.k)
		)

		# Tokenize each kmer
		return self.tokenizer.encode_plus(
			kmers.tolist(),
			is_split_into_words=True,
			return_tensors='pt',
		)

	def _get_item_bert(self, item_data):
		pre_seq_string = item_data['seq_string'][:self.n_per_side]
		post_seq_string = item_data['seq_string'][-self.n_per_side:]

		pre_feats = self.tokenize_seq_str(pre_seq_string)
		post_feats = self.tokenize_seq_str(post_seq_string)

		if self.is_binary:
			return {
				'pre_input_ids': pre_feats['input_ids'][0],
				'pre_token_type_ids': pre_feats['token_type_ids'][0],
				'pre_attention_mask': pre_feats['attention_mask'][0],
				'post_input_ids': post_feats['input_ids'][0],
				'post_token_type_ids': post_feats['token_type_ids'][0],
				'post_attention_mask': post_feats['attention_mask'][0],
				'pre_seq_string': pre_seq_string,
				'post_seq_string': post_seq_string,
				'chr_loc': item_data['chr_loc'],
				'label': torch.tensor(item_data['label'] > 0).float(),
				'label_val': torch.tensor(item_data['label']).float()
			}
		else:
			return {
				'pre_input_ids': pre_feats['input_ids'][0],
				'pre_token_type_ids': pre_feats['token_type_ids'][0],
				'pre_attention_mask': pre_feats['attention_mask'][0],
				'post_input_ids': post_feats['input_ids'][0],
				'post_token_type_ids': post_feats['token_type_ids'][0],
				'post_attention_mask': post_feats['attention_mask'][0],
				'pre_seq_string': pre_seq_string,
				'post_seq_string': post_seq_string,
				'chr_loc': item_data['chr_loc'],
				'label': torch.tensor(item_data['label']).float()
			}

	def __getitem__(self, idx):
		""" Return pre- and post-sequence features is 5-feat or BERT format. """
		item_data = self.split_data[idx]

		if self.tokenizer is None:
			return self._get_item_5_feat_mat(item_data)
		else:
			return self._get_item_bert(item_data)
		


class STRHetPrePostDataModule(STRDataModule):
	"""Takes labels from regression and makes into >0 classification problem."""
	def __init__(self, data_dir, split_file, batch_size = 32, num_workers = 1,
					shuffle = True, n_per_side = 500, tokenizer = None, k = 5,
					is_binary=True):
		super().__init__(data_dir, split_file, batch_size, num_workers, shuffle)
		self.n_per_side = n_per_side
		self.tokenizer = tokenizer
		self.k = k
		self.is_binary = is_binary

	def setup(self, stage=None):
		self.train_data = STRHetPrePostDataset(
			self.data_dir, self.split_file, 'train', self.n_per_side,
			self.tokenizer, self.k, self.is_binary
		)
		self.val_data = STRHetPrePostDataset(
			self.data_dir, self.split_file, 'val', self.n_per_side,
			self.tokenizer, self.k, self.is_binary
		)
		self.test_data = STRHetPrePostDataset(
			self.data_dir, self.split_file, 'test', self.n_per_side,
			self.tokenizer, self.k, self.is_binary
		)


if __name__ == "__main__":
	from dnabert.transformers.configuration_bert import BertConfig
	from dnabert.transformers.modeling_bert import BertForSequenceClassification
	from dnabert.transformers.tokenization_bert import BertTokenizer

	# Testing
	# data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples2')
	# split_file = 'split_1.json'

	# STR_data = STRDataModule(data_dir, split_file, num_workers=2)
	# STR_data.setup()

	# print(next(iter(STR_data.val_dataloader()))['feat_mat'].shape)

	# STR_data_nz = STRDataModuleZonzeroClass(data_dir, split_file, num_workers=2)
	# STR_data_nz.setup()

	# print(next(iter(STR_data_nz.val_dataloader()))['feat_mat'].shape)

	# Load BERT
	bert_config = BertConfig.from_pretrained(
		'dnabert/5-new-12w-0/',
		num_labels=2
	)
	tokenizer = BertTokenizer.from_pretrained(
		'dnabert/5-new-12w-0/',
		from_tf=False,
		do_lower_case=False
	)
	model = BertForSequenceClassification.from_pretrained(
		'dnabert/5-new-12w-0/',
		from_tf=False,
		config=bert_config,
	)

	# STR het prepost
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	split_file = 'split_1.json'

	bert_dataset = STRHetPrePostDataset(
		data_dir, 
		split_file, 
		'val', 
		n_per_side=500,
		tokenizer=tokenizer,
		k=5	
	)

	STR_data = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		num_workers=2, 
		n_per_side=500,
		tokenizer=tokenizer,
		k=5	
	)
	STR_data.setup()
	# print(next(iter(STR_data.val_dataloader()))['pre_feat_mat'].shape)
	# print(next(iter(STR_data.val_dataloader()))['post_feat_mat'].shape)
	print(STR_data.val_data[0])

	d = STR_data.val_data[0]

	model_out = model(
		input_ids=d['pre_input_ids'],
		token_type_ids=d['pre_token_type_ids'],
		attention_mask=d['pre_attention_mask'],
	)