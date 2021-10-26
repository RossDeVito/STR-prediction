"""Load trained models and compare performance."""

import os

import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics import ConfusionMatrix, Precision, Recall, F1, PrecisionRecallCurve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from data_modules import STRDataModuleZonzeroClass
from models import STRClassifier, ResNet


if __name__ == '__main__':
	models_to_eval = [
		{
			'name': 'ResNet 1, lr=.001',
			'model': ResNet(),
			'load_path': '../trained_models/heterozygosity_bin/resnet_1_lr001/checkpoints/epoch=8-best_val_loss.ckpt'
		},
		{   'name': 'ResNet 1, lr=.00025',
			'model': ResNet(),
			'load_path': '../trained_models/heterozygosity_bin/resnet_1_lr00025/checkpoints/epoch=8-best_val_loss.ckpt'
		},
	]

	# Load data splits
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples2')
	split_file = 'split_1.json'

	data_mod = STRDataModuleZonzeroClass(
		data_dir, 
		split_file, 
		batch_size=256,
		num_workers=3
	)
	data_mod.setup()

	# Make predictions and evaluate with trained each model
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

	for m in models_to_eval:
		model = STRClassifier.load_from_checkpoint(
			m['load_path'],
			model=m['model']
		)
		trainer = pl.Trainer(weights_summary=None, gpus=1)

		preds = trainer.predict(
			model=model, 
			dataloaders=data_mod.val_dataloader()
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
			y_true.int().numpy(), y_pred.numpy()
		)
		res_dict['PR_auc'] = auc(res_dict['PR_recall'], res_dict['PR_prec'])
		
		results.append(res_dict)

	# plot ROC and PR curves
	lw = 1

	fig, (ax_roc, ax_pr) = plt.subplots(1, 2, sharex=True, sharey=True, 
										subplot_kw=dict(box_aspect=1))

	for res in results:
		ax_roc.plot(res['ROC_fpr'], res['ROC_tpr'], label=res['name'], lw=lw)
		ax_pr.plot(res['PR_recall'], res['PR_prec'], label=res['name'], lw=lw)

	fig.suptitle("Binary Heterozygosity Prediction")
	ax_roc.set_title("ROC")
	ax_roc.set_xlabel("False Positive Rate")
	ax_roc.set_ylabel("True Positive Rate")
	ax_roc.legend()
	ax_pr.set_title("PR Curve")
	ax_pr.set_xlabel("Recall")
	ax_pr.set_ylabel("Precision")
	ax_pr.legend()

	fig.savefig('../trained_models/heterozygosity_bin/roc_pr.png')
	plt.close(fig)

	# plot confusion matrices
	fig, axs = plt.subplots(ncols=len(results), subplot_kw=dict(box_aspect=1))

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
	fig.tight_layout()
	fig.savefig('../trained_models/heterozygosity_bin/CMs.png')
	plt.close(fig)