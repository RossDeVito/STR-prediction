"""Load trained models and compare performance."""

import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics import ConfusionMatrix, Precision, Recall, F1, PrecisionRecallCurve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from data_modules import STRDataModule
from models import STRPrePostClassifier, PrePostModel, InceptionPrePostModel

import cnn_models


if __name__ == '__main__':
	__spec__ = None

	num_feat_channels = 6
	num_gpu = 0

	# models_to_eval = [
	# 	{   'name': 'Inception v1 (3,1), bs512',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=3,
	# 			n_filters_fe=32,
	# 			depth_pred=1,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[3, 5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.2
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round1/version_0_bs512/checkpoints/epoch=33-best_val_loss.ckpt'
	# 	},
	# 	# {   'name': 'Inception v1 (3,1), bs512 last',
	# 	# 	'model': InceptionPrePostModel(
	# 	# 		in_channels=num_feat_channels,
	# 	# 		depth_fe=3,
	# 	# 		n_filters_fe=32,
	# 	# 		depth_pred=1,
	# 	# 		n_filters_pred=32,
	# 	# 		kernel_sizes=[3, 5, 9, 19],
	# 	# 		activation='gelu',
	# 	# 		dropout=.2
	# 	# 	),
	# 	# 	'load_path': '../trained_models/het_cls_v2/round1/version_0_bs512/checkpoints/epoch=83-last.ckpt'
	# 	# },
	# 	{   'name': 'Inception v1 (3,1), bs256',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=3,
	# 			n_filters_fe=32,
	# 			depth_pred=1,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[3, 5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.2
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round1/version_1/checkpoints/epoch=25-best_val_loss.ckpt'
	# 	},
	# 	# {   'name': 'Inception v1 (3,1), bs256 last',
	# 	# 	'model': InceptionPrePostModel(
	# 	# 		in_channels=num_feat_channels,
	# 	# 		depth_fe=3,
	# 	# 		n_filters_fe=32,
	# 	# 		depth_pred=1,
	# 	# 		n_filters_pred=32,
	# 	# 		kernel_sizes=[3, 5, 9, 19],
	# 	# 		activation='gelu',
	# 	# 		dropout=.2
	# 	# 	),
	# 	# 	'load_path': '../trained_models/het_cls_v2/round1/version_1/checkpoints/epoch=75-last.ckpt'
	# 	# },
	# 	{   'name': 'Inception v1 (6,3), bs256',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=6,
	# 			n_filters_fe=32,
	# 			depth_pred=3,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[3, 5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.2
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round1/version_2/checkpoints/epoch=29-best_val_loss.ckpt'
	# 	},
	# 	# {   'name': 'Inception v1 (6,3), bs256 last',
	# 	# 	'model': InceptionPrePostModel(
	# 	# 		in_channels=num_feat_channels,
	# 	# 		depth_fe=6,
	# 	# 		n_filters_fe=32,
	# 	# 		depth_pred=3,
	# 	# 		n_filters_pred=32,
	# 	# 		kernel_sizes=[3, 5, 9, 19],
	# 	# 		activation='gelu',
	# 	# 		dropout=.2
	# 	# 	),
	# 	# 	'load_path': '../trained_models/het_cls_v2/round1/version_2/checkpoints/epoch=79-last.ckpt'
	# 	# },
	# 	{   'name': 'Inception v1 (6,3), bs256',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=6,
	# 			n_filters_fe=32,
	# 			depth_pred=3,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[3, 5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.2
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round1/version_2/checkpoints/epoch=29-best_val_loss.ckpt'
	# 	},
	# 	# {   'name': 'Inception v1 (6,3), bs256 last',
	# 	# 	'model': InceptionPrePostModel(
	# 	# 		in_channels=num_feat_channels,
	# 	# 		depth_fe=6,
	# 	# 		n_filters_fe=32,
	# 	# 		depth_pred=3,
	# 	# 		n_filters_pred=32,
	# 	# 		kernel_sizes=[3, 5, 9, 19],
	# 	# 		activation='gelu',
	# 	# 		dropout=.2
	# 	# 	),
	# 	# 	'load_path': '../trained_models/het_cls_v2/round1/version_2/checkpoints/epoch=79-last.ckpt'
	# 	# },
	# 	{   'name': 'Inception v1 (6,3), bs256 do.3',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=6,
	# 			n_filters_fe=32,
	# 			depth_pred=3,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[3, 5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.3
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round1/version_5/checkpoints/epoch=80-best_val_loss.ckpt'
	# 	},
	# 	{   'name': 'Inception v1 (6,3), bs256 do.3 last',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=6,
	# 			n_filters_fe=32,
	# 			depth_pred=3,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[3, 5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.3
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round1/version_5/checkpoints/epoch=124-last.ckpt'
	# 	},
	# ]
	## Round2
	# models_to_eval = [
	# 	{   'name': 'Inception v1 (3,1), bs256',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=3,
	# 			n_filters_fe=32,
	# 			depth_pred=1,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.3
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round2/version_0/checkpoints/epoch=47-best_val_loss.ckpt'
	# 	},
	# 	{   'name': 'Inception v1 (3,1), bs256 last',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=3,
	# 			n_filters_fe=32,
	# 			depth_pred=1,
	# 			n_filters_pred=32,
	# 			kernel_sizes=[5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.3
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round2/version_0/checkpoints/epoch=97-last.ckpt'
	# 	},
	# 	{   'name': 'Inception v1 (5,2), bs256',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=5,
	# 			n_filters_fe=64,
	# 			depth_pred=2,
	# 			n_filters_pred=64,
	# 			kernel_sizes=[5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.3
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round2/version_1/checkpoints/epoch=65-best_val_loss.ckpt'
	# 	},
	# 	{   'name': 'Inception v1 (5,2), bs256 last',
	# 		'model': InceptionPrePostModel(
	# 			in_channels=num_feat_channels,
	# 			depth_fe=5,
	# 			n_filters_fe=64,
	# 			depth_pred=2,
	# 			n_filters_pred=64,
	# 			kernel_sizes=[5, 9, 19],
	# 			activation='gelu',
	# 			dropout=.3
	# 		),
	# 		'load_path': '../trained_models/het_cls_v2/round2/version_1/checkpoints/epoch=115-last.ckpt'
	# 	},
	# ]
	## Round3
	models_to_eval = [
		{   'name': 'Inception 0 ((3,32),(1,32)) [5,9,19]',
			'model': InceptionPrePostModel(
				in_channels=num_feat_channels,
				depth_fe=3,
				n_filters_fe=32,
				depth_pred=1,
				n_filters_pred=32,
				kernel_sizes=[5, 9, 19],
				activation='gelu',
				dropout=.3
			),
			'load_path': '../trained_models/het_cls_v2/round3/V2_1/version_0/checkpoints/epoch=47-best_val_loss.ckpt'
		},
		{   'name': 'Inception 1 ((5,64),(2,64)) [5,9,19]',
			'model': InceptionPrePostModel(
				in_channels=num_feat_channels,
				depth_fe=5,
				n_filters_fe=64,
				depth_pred=2,
				n_filters_pred=64,
				kernel_sizes=[5, 9, 19],
				activation='gelu',
				dropout=.3
			),
			'load_path': '../trained_models/het_cls_v2/round3/V2_1/version_1/checkpoints/epoch=65-best_val_loss.ckpt'
		},
		{   'name': 'Inception 2 ((5,64),(2,64)) [5,9,19] ws256',
			'model': InceptionPrePostModel(
				in_channels=num_feat_channels,
				depth_fe=5,
				n_filters_fe=64,
				depth_pred=2,
				n_filters_pred=64,
				kernel_sizes=[5, 9, 19],
				activation='gelu',
				dropout=.3
			),
			'load_path': '../trained_models/het_cls_v2/round3/V2_1/version_2/checkpoints/epoch=15-best_val_loss.ckpt'
		},
		{   'name': 'Inception 3 ((5,64),(2,64)) [5,9,19] lr1e-5',
			'model': InceptionPrePostModel(
				in_channels=num_feat_channels,
				depth_fe=5,
				n_filters_fe=64,
				depth_pred=2,
				n_filters_pred=64,
				kernel_sizes=[5, 9, 19],
				activation='gelu',
				dropout=.3
			),
			'load_path': '../trained_models/het_cls_v2/round3/V2_1/version_3/checkpoints/epoch=22-best_val_loss.ckpt'
		},
		{   'name': 'Inception 4 ((5,64),(2,64)) [5,9,19] lr1e-3',
			'model': InceptionPrePostModel(
				in_channels=num_feat_channels,
				depth_fe=5,
				n_filters_fe=64,
				depth_pred=2,
				n_filters_pred=64,
				kernel_sizes=[5, 9, 19],
				activation='gelu',
				dropout=.3
			),
			'load_path': '../trained_models/het_cls_v2/round3/V2_1/version_4/checkpoints/epoch=12-best_val_loss.ckpt'
		},
	]

	# Metrics for each model
	metrics = MetricCollection({
		'macro_precision': Precision(num_classes=2, average='macro', multiclass=True),
		'class_precision': Precision(num_classes=2, average='none', multiclass=True),
		'macro_recall': Recall(num_classes=2, average='macro', multiclass=True),
		'class_recall': Recall(num_classes=2, average='none', multiclass=True),
		'macro_F1': F1(num_classes=2, average='macro', multiclass=True),
		'class_F1': F1(num_classes=2, average='none', multiclass=True),
		'confusion_matrix': ConfusionMatrix(num_classes=2)
	})
	results = []

	# Eval models
	training_params = {
		# Data Module
		'batch_size': 512,
		'min_copy_number': None,
		'max_copy_number': 15,
		'incl_STR_feat': True,
		'min_boundary_STR_pos': 6,
		'max_boundary_STR_pos': 6,
		'window_size': 128,
		'bp_dist_units': 1000.0,
		'split_name': 'split_1',
	}
	num_workers_per_loader = 3

	data_path = os.path.join(
		'..', 'data', 'heterozygosity', 'sample_data_V2_repeat_var.json'
	)
	data_mod = STRDataModule(
		data_path,
		split_name=training_params['split_name'],
		batch_size=training_params['batch_size'],
		num_workers=num_workers_per_loader,
		incl_STR_feat=training_params['incl_STR_feat'],
		min_boundary_STR_pos=training_params['min_boundary_STR_pos'],
		max_boundary_STR_pos=training_params['max_boundary_STR_pos'],
		window_size=training_params['window_size'],
		min_copy_num=training_params['min_copy_number'],
		max_copy_num=training_params['max_copy_number'],
		bp_dist_units=training_params['bp_dist_units']
	)
	data_mod.setup()

	for m in models_to_eval:
		model = STRPrePostClassifier.load_from_checkpoint(
			m['load_path'],
			model=m['model']
		)
		trainer = pl.Trainer(gpus=num_gpu)

		preds = trainer.predict(
			model=model, 
			dataloaders=data_mod.test_dataloader()
		)

		y_pred = torch.cat([r['y_hat'].cpu() for r in preds])
		y_true = torch.cat([r['y_true'].cpu() for r in preds])

		res_dict = metrics(y_pred, y_true.int())

		res_dict['name'] = m['name']
		res_dict['num_true_0'] = (y_true == 0).sum().item()
		res_dict['num_true_1'] = (y_true == 1).sum().item()

		res_dict['ROC_fpr'], res_dict['ROC_tpr'], _ = roc_curve(
			y_true.int().numpy(), y_pred.numpy()
		)
		res_dict['ROC_auc'] = auc(res_dict['ROC_fpr'], res_dict['ROC_tpr'])

		res_dict['PR_prec'], res_dict['PR_recall'], _ = precision_recall_curve(
			y_true.int().numpy(), y_pred.numpy(), #pos_label=0
		)
		res_dict['PR_auc'] = auc(res_dict['PR_recall'], res_dict['PR_prec'])
		
		results.append(res_dict)


	# plot ROC and PR curves
	lw = 2

	fig, (ax_roc, ax_pr) = plt.subplots(1, 2, sharex=True, sharey=True, 
										subplot_kw=dict(box_aspect=1),
										figsize=(12, 6))
	ax_roc.plot([0, 1], [0, 1], color="black", lw=lw, linestyle=":")

	for res in results:
		ax_roc.plot(res['ROC_fpr'], res['ROC_tpr'], label=res['name'], lw=lw,
					linestyle='--')
		ax_pr.plot(res['PR_recall'], res['PR_prec'], label=res['name'], lw=lw,
					linestyle='--')

	fig.suptitle("Binary Heterozygosity Prediction")
	ax_roc.set_title("ROC")
	ax_roc.set_xlabel("False Positive Rate")
	ax_roc.set_ylabel("True Positive Rate")
	ax_roc.legend(loc='lower right')
	ax_pr.set_title("PR Curve")
	ax_pr.set_xlabel("Recall")
	ax_pr.set_ylabel("Precision")
	ax_pr.legend(loc='lower left')
	fig.tight_layout()

	fig.savefig('../trained_models/het_cls_v2/round3/V2_1/roc_pr.png',
				bbox_inches='tight')
	plt.close(fig)

	# plot confusion matrices
	unit_len = 3
	fig, axs = plt.subplots(ncols=len(results), subplot_kw=dict(box_aspect=1),
				figsize=(len(results)*unit_len, unit_len))

	for i, res in enumerate(results):
		axs[i].set_title(res['name'])
		sns.heatmap(
			res['confusion_matrix'], 
			annot=True,
			fmt='0.0f',
			cbar=False, 
			ax=axs[i],
		)

	axs[0].set_ylabel("True")
	axs[0].set_xlabel("Pred")
	# fig.tight_layout()
	fig.savefig('../trained_models/het_cls_v2/round3/V2_1/CMs.png',
				bbox_inches='tight')
	plt.close(fig)

	# Make table
	df = pd.DataFrame(results)

	table = df[['name', 'macro_F1', 'ROC_auc', 'class_precision', 'class_recall']]
	table['macro_F1'] = [s.item() for s in table.macro_F1]
	# table['ROC_auc'] = [s.item() for s in table.ROC_auc]
	table['class_precision_0'] = [s[0].item() for s in table.class_precision]
	table['class_precision_1'] = [s[1].item() for s in table.class_precision]
	table['class_recall_0'] = [s[0].item() for s in table.class_recall]
	table['class_recall_1'] = [s[1].item() for s in table.class_recall]
	table = table.drop(columns=['class_precision', 'class_recall'])
	table.to_csv('../trained_models/het_cls_v2/round3/V2_1/table.csv')

	print(table)