"""Score a model by specifying dir and save results there."""

import os
import json
import platform

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics import ConfusionMatrix, Precision, Recall, F1, PrecisionRecallCurve
from sklearn.metrics import roc_curve, auc, precision_recall_curve

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

	# Select model's output dir
	if motif_type == '5comb':
		trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1_AC-AG-AT-CT-GT'
	elif motif_type == 'AC':
		trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1'
	elif motif_type == 'T':
		trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1_T'
	else:
		raise ValueError("Invalid motif_type: {}".format(motif_type))
	model_dir = 'version_10'

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

	# Metrics
	metrics = MetricCollection({
		'macro_precision': Precision(num_classes=2, average='macro', multiclass=True),
		'class_precision': Precision(num_classes=2, average='none', multiclass=True),
		'macro_recall': Recall(num_classes=2, average='macro', multiclass=True),
		'class_recall': Recall(num_classes=2, average='none', multiclass=True),
		'macro_F1': F1(num_classes=2, average='macro', multiclass=True),
		'class_F1': F1(num_classes=2, average='none', multiclass=True),
		'confusion_matrix': ConfusionMatrix(num_classes=2)
	})

	# Make predictions and score
	trainer = pl.Trainer(gpus=num_gpus)
	preds = trainer.predict(
		model=model, 
		dataloaders=data_module.test_dataloader()
	)
	y_pred = torch.cat([r['y_hat'].cpu() for r in preds])
	y_true = torch.cat([r['y_true'].cpu() for r in preds])

	res_dict = metrics(y_pred, y_true.int())
	# make numpy, so can then be turned into a list before saving as JSON
	res_dict = {k: v.numpy() for k, v in res_dict.items()}
	res_dict['y_true'] = y_true.numpy()
	res_dict['y_pred'] = y_pred.numpy()

	res_dict['num_true_0'] = (y_true == 0).sum().item()
	res_dict['num_true_1'] = (y_true == 1).sum().item()

	res_dict['ROC_fpr'], res_dict['ROC_tpr'], _ = roc_curve(
		y_true.int().numpy(), y_pred.numpy()
	)
	res_dict['ROC_auc'] = auc(res_dict['ROC_fpr'], res_dict['ROC_tpr'])

	res_dict['PR_prec_1'], res_dict['PR_recall_1'], _ = precision_recall_curve(
		y_true.int().numpy(), y_pred.numpy(), pos_label=1
	)
	res_dict['PR_auc_1'] = auc(res_dict['PR_recall_1'], res_dict['PR_prec_1'])

	res_dict['PR_prec_0'], res_dict['PR_recall_0'], _ = precision_recall_curve(
		y_true.int().numpy(), -y_pred.numpy(), pos_label=0
	)
	res_dict['PR_auc_0'] = auc(res_dict['PR_recall_0'], res_dict['PR_prec_0'])

	# Save results as lists
	res_dict = {
		k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in res_dict.items() 
	}

	# Save results
	res_save_dir = os.path.join(trained_res_dir, model_dir, 'results')
	if not os.path.exists(res_save_dir):
		os.makedirs(res_save_dir)
	res_save_file = os.path.join(
		res_save_dir, 
		'results_{}.json'.format('best' if use_best_loss else 'last')
	)
	with open(res_save_file, 'w') as f:
		json.dump(res_dict, f, indent=2)