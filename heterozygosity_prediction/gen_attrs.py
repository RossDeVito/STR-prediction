import os
import json
import platform
import pickle

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from captum import attr
from tqdm import tqdm

from model_utils import count_params
from data_modules import STRDataModule
import models
import prepost_models


if __name__ == '__main__':
	__spec__ = None
	if platform.system() == 'Darwin':
		num_gpus = 0
		print("Running on MacOS, setting num_gpus to 0")
	else:
		num_gpus = 1

	# General options
	num_workers_per_loader = 3
	motif_type = ['T', 'AC', '5comb'][2]
	integrated_gradients_batch_size = 1024

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

	# whether to use best val loss or last epoch
	use_best_loss = True

	# Load model params
	with open(os.path.join(trained_res_dir, model_dir, 'train_params.json'), 'r') as f:
		model_params = json.load(f)

	if 'model_type' not in model_params.keys():
		model_params['model_type'] = 'InceptionPrePostModel'

	# Load data
	if motif_type == 'AC':
		data_path = os.path.join(
			'..', 'data', 'heterozygosity', 'sample_data_V2_repeat_var.json'
		)
	elif motif_type == 'T':
		data_path = os.path.join(
			'..', 'data', 'heterozygosity', 'sample_data_T_V2_repeat_var.json'
		)
	elif motif_type == '5comb':
		data_path = os.path.join(
			'..', 'data', 'heterozygosity', 'sample_data_AC-AG-AT-CT-GT_V2_repeat_var.json'
		)
	data_module = STRDataModule(
		data_path,
		return_data=True, 
		return_strings=True,
		split_name=model_params['split_name'],
		batch_size=model_params['batch_size'],
		num_workers=num_workers_per_loader,
		incl_STR_feat=model_params['incl_STR_feat'],
		min_boundary_STR_pos=model_params['min_boundary_STR_pos'],
		max_boundary_STR_pos=model_params['max_boundary_STR_pos'],
		window_size=model_params['window_size'],
		min_copy_num=model_params['min_copy_number'],
		max_copy_num=model_params['max_copy_number'],
		bp_dist_units=model_params['bp_dist_units']
	)
	data_module.setup()

	# Create model
	if model_params['model_type'] == 'InceptionPrePostModel':
		net = prepost_models.InceptionPrePostModel(
			in_channels=data_module.num_feat_channels(),
			depth_fe=model_params['depth_fe'],
			n_filters_fe=model_params['n_filters_fe'],
			depth_pred=model_params['depth_pred'],
			n_filters_pred=model_params['n_filters_pred'],
			kernel_sizes=model_params['kernel_sizes'],
			activation=model_params['activation'],
			dropout=model_params['dropout']
		)
	elif model_params['model_type'] == 'InceptionPreDimRedPost':
		net = prepost_models.InceptionPreDimRedPost(
			n_per_side=model_params['window_size'],
			reduce_to=model_params['reduce_to'],
			in_channels=data_module.num_feat_channels(),
			depth_fe=model_params['depth_fe'],
			pool_size=model_params['pool_size'],
			n_filters_fe=model_params['n_filters_fe'],
			kernel_sizes_fe=model_params['kernel_sizes'],
			kernel_sizes_pred=model_params['kernel_sizes_pred'],
			n_filters_pred=model_params['n_filters_pred'],
			activation=model_params['activation'],
			dropout_cnn=model_params['dropout'],
			dropout_dense=model_params['dropout_dense'],
			dense_layer_sizes=model_params['dense_layer_sizes']
		)
	else:
		raise ValueError("Unknown model type: {}".format(model_params['model_type']))
	
	if 'model_path' in model_params.keys():
		print(model_params['model_n_params'], count_params(net))
		assert model_params['model_n_params'] == count_params(net)
	else:
		print(count_params(net))

	# Load model weights
	saved_weights = os.listdir(os.path.join(trained_res_dir, model_dir, 'checkpoints'))
	if use_best_loss:
		weights_file = [f for f in saved_weights if 'best_val_loss.ckpt' in f]
		assert len(weights_file) == 1
		weights_file = weights_file[0]
	else:
		weights_file = [f for f in saved_weights if '-last.ckpt' in f]
		assert len(weights_file) == 1
		weights_file = weights_file[0]
	
	weights_path = os.path.join(
		trained_res_dir, model_dir, 'checkpoints', weights_file
	)

	model = models.STRPrePostClassifier.load_from_checkpoint(
		weights_path,
		model=net,
	)
	model.eval()
	if num_gpus > 0:
		model.cuda()

	# Setup attr methods
	attr_methods = {
		# 'saliency': attr.Saliency(model),
		# 'gbp': attr.GuidedBackprop(model),
		# 'deconv': attr.Deconvolution(model),
		'ig_global': attr.IntegratedGradients(model, multiply_by_inputs=True),
		# 'ig_local': attr.IntegratedGradients(model, multiply_by_inputs=False),
		# 'deep_lift': attr.DeepLift(model, multiply_by_inputs=True),
		# 'deep_list_shap': attr.DeepLiftShap(model, multiply_by_inputs=True),
		# 'grad_shap': attr.GradientShap(model, multiply_by_inputs=True),
	}
	data_splits = [
		'train',
		'val',
		'test',
	]
	attrs_dict = dict()
	for method in attr_methods.keys():
		attrs_dict[method] = dict()
		attrs_dict[method]['pre'] = []
		attrs_dict[method]['post'] = []

	# Data to save
	predictions = []
	labels = []
	sample_data = []
	pre_strings = []
	post_strings = []
	split = []
	
	# Generate attributions with captum
	if 'train' in data_splits:
		dl = data_module.train_dataloader()
		for i, batch in tqdm(enumerate(dl), total=len(dl), desc='train set'):
			if num_gpus > 0:
				batch_feats = (
					batch.pop('pre_STR_feats').cuda(),
					batch.pop('post_STR_feats').cuda()
				)
			else:
				batch_feats = (
					batch.pop('pre_STR_feats'),
					batch.pop('post_STR_feats')
				)
			
			# Get predictions and sample data
			predictions.extend(model(*batch_feats).flatten().tolist())
			labels.extend(batch['label'].tolist())
			sample_data.extend(batch['data'])
			pre_strings.extend(batch['pre_STR_seq'])
			post_strings.extend(batch['post_STR_seq'])
			split.extend(['train'] * len(batch['label']))

			# Get attributions
			for method_name, attr_module in attr_methods.items():
				print(method_name)
				if method_name in ['ig_global', 'ig_local']:
					attr_vals = attr_module.attribute(
						batch_feats,
						internal_batch_size=integrated_gradients_batch_size
					)
				else:
					attr_vals = attr_module.attribute(batch_feats)

				attrs_dict[method_name]['pre'].extend(
					attr_vals[0].detach().cpu().numpy()
				)
				attrs_dict[method_name]['post'].extend(
					attr_vals[1].detach().cpu().numpy()
				)
	if 'val' in data_splits:
		dl = data_module.val_dataloader()
		for i, batch in tqdm(enumerate(dl), total=len(dl), desc='val set'):
			if num_gpus > 0:
				batch_feats = (
					batch.pop('pre_STR_feats').cuda(),
					batch.pop('post_STR_feats').cuda()
				)
			else:
				batch_feats = (
					batch.pop('pre_STR_feats'),
					batch.pop('post_STR_feats')
				)
			
			# Get predictions and sample data
			predictions.extend(model(*batch_feats).flatten().tolist())
			labels.extend(batch['label'].tolist())
			sample_data.extend(batch['data'])
			pre_strings.extend(batch['pre_STR_seq'])
			post_strings.extend(batch['post_STR_seq'])
			split.extend(['val'] * len(batch['label']))

			# Get attributions
			for method_name, attr_module in attr_methods.items():
				print(method_name)
				if method_name in ['ig_global', 'ig_local']:
					attr_vals = attr_module.attribute(
						batch_feats,
						internal_batch_size=integrated_gradients_batch_size
					)
				else:
					attr_vals = attr_module.attribute(batch_feats)

				attrs_dict[method_name]['pre'].extend(
					attr_vals[0].detach().cpu().numpy()
				)
				attrs_dict[method_name]['post'].extend(
					attr_vals[1].detach().cpu().numpy()
				)
	if 'test' in data_splits:
		dl = data_module.test_dataloader()
		for i, batch in tqdm(enumerate(dl), total=len(dl), desc='test set'):
			if num_gpus > 0:
				batch_feats = (
					batch.pop('pre_STR_feats').cuda(),
					batch.pop('post_STR_feats').cuda()
				)
			else:
				batch_feats = (
					batch.pop('pre_STR_feats'),
					batch.pop('post_STR_feats')
				)
			
			# Get predictions and sample data
			predictions.extend(model(*batch_feats).flatten().tolist())
			labels.extend(batch['label'].tolist())
			sample_data.extend(batch['data'])
			pre_strings.extend(batch['pre_STR_seq'])
			post_strings.extend(batch['post_STR_seq'])
			split.extend(['test'] * len(batch['label']))

			# Get attributions
			for method_name, attr_module in attr_methods.items():
				print(method_name)
				if method_name in ['ig_global', 'ig_local']:
					attr_vals = attr_module.attribute(
						batch_feats,
						internal_batch_size=integrated_gradients_batch_size
					)
				else:
					attr_vals = attr_module.attribute(batch_feats)

				attrs_dict[method_name]['pre'].extend(
					attr_vals[0].detach().cpu().numpy()
				)
				attrs_dict[method_name]['post'].extend(
					attr_vals[1].detach().cpu().numpy()
				)

	# Save attributions with associated data as pickles
	res_save_dir = os.path.join(trained_res_dir, model_dir, 'attributions')
	if not os.path.exists(res_save_dir):
		os.makedirs(res_save_dir)

	for method_name, attrs in attrs_dict.items():
		attrs['predictions'] = predictions
		attrs['labels'] = labels
		attrs['sample_data'] = sample_data
		attrs['pre_strings'] = pre_strings
		attrs['post_strings'] = post_strings

		with open(os.path.join(res_save_dir, '{}.pkl'.format(method_name)), 'wb') as f:
			pickle.dump(attrs, f)

	


	