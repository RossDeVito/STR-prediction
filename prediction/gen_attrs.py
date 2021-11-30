import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from captum import attr

from data_modules import STRHetPrePostDataModule
from models import STRPrePostClassifier, PrePostModel, InceptionPrePostModel


if __name__ == '__main__':
	use_gpu = torch.cuda.is_available()
	save_dir = os.path.join('..', 'attr_data', 'heterozygosity', 
		'incep_v1_4_2_bs256', 'saliency')

	# Load model
	model_details = {   
		'name': 'Inception v1 (4,2), bs256 last',
		'model': InceptionPrePostModel(),
		'load_path': '../trained_models/heterozygosity_bin_prepost/incep_v1_4_2_bs256/checkpoints/epoch=50-last.ckpt'
	}
	model = STRPrePostClassifier.load_from_checkpoint(
		model_details['load_path'],
		model=model_details['model']
	)
	model.eval()
	if use_gpu:
		model.cuda()

	# Load data
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	split_file = 'split_1.json'

	data_mod = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		batch_size=2,
		num_workers=3
	)
	data_mod.setup()
	data_loader = data_mod.test_dataloader()

	# Setup attr method
	saliency = attr.Saliency(model)

	# Data to save
	pre_strings = []
	post_strings = []
	pre_attrs = []
	post_attrs = []
	chr_locs = []
	labels = []
	label_vals = []
	preds_list = []

	# Use captum to generate attributions
	for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
		if use_gpu:
			batch_feats = (
				batch.pop('pre_feat_mat').cuda(),
				batch.pop('post_feat_mat').cuda()
			)
		else:
			batch_feats = (
				batch.pop('pre_feat_mat'),
				batch.pop('post_feat_mat')
			)

		# Get attributions
		grads = saliency.attribute(batch_feats, target=0)
		preds = model(*batch_feats).flatten().tolist()

		# Store data and results
		pre_strings.extend(batch['pre_seq_string'])
		post_strings.extend(batch['post_seq_string'])
		pre_attrs.append(grads[0].cpu().numpy())
		post_attrs.append(grads[1].cpu().numpy())
		chr_locs.extend(batch['chr_loc'])
		labels.extend(batch['label'].tolist())
		label_vals.extend(batch['label_val'].tolist())
		preds_list.extend(preds)
		
	# Save data
	meta_df = pd.DataFrame({
		'pred': preds_list,
		'label': labels,
		'label_val': label_vals,
		'chr_loc': chr_locs,
		'pre_string': pre_strings,
		'post_string': post_strings,
	})
	meta_df.to_csv(os.path.join(save_dir, 'meta.csv'), index=False)

	pre_attrs = np.concatenate(pre_attrs, axis=0)
	np.save(os.path.join(save_dir, 'pre_attrs.npy'), pre_attrs)

	post_attrs = np.concatenate(post_attrs, axis=0)
	np.save(os.path.join(save_dir, 'post_attrs.npy'), post_attrs)