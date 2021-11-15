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

from data_modules import STRHetPrePostDataModule
from models import STRPrePostClassifier, PrePostModel, InceptionPrePostModel
from models import ConcatPredictorBert
from feature_extractors import DNABERT

from dnabert.transformers.tokenization_bert import BertTokenizer


def make_DNABERT_model(bert_save_dir):
	fe = DNABERT(bert_save_dir)
	cls_module = ConcatPredictorBert()
	return PrePostModel(feature_extractor=fe, predictor=cls_module) 


if __name__ == '__main__':
	models_to_eval = [
		{   'name': 'Inception v1 (4,2), bs128',
			'model': InceptionPrePostModel(),
			'load_path': '../trained_models/heterozygosity_bin_prepost/incep_v1_4_2_bs128/checkpoints/epoch=17-best_val_loss.ckpt'
		},
		{   'name': 'Inception v1 (4,2), bs256',
			'model': InceptionPrePostModel(),
			'load_path': '../trained_models/heterozygosity_bin_prepost/incep_v1_4_2_bs256/checkpoints/epoch=25-best_val_loss.ckpt'
		},
	]

	bert_save_dir = 'dnabert/5-new-12w-0/'
	bert_models_to_eval = [
		# {
		# 	'name': 'DNABERT bs2',
		# 	'model': make_DNABERT_model(bert_save_dir),
		# 	'load_path': '../trained_models/heterozygosity_bin_prepost/dnabert_v1_bs2/checkpoints/epoch=2-best_val_loss.ckpt'
		# },
		{
			'name': 'DNABERT bs4',
			'model': make_DNABERT_model(bert_save_dir),
			'load_path': '../trained_models/heterozygosity_bin_prepost/dnabert_v1_bs4/checkpoints/epoch=2-best_val_loss.ckpt'
		},
		{
			'name': 'DNABERT v6',
			'model': make_DNABERT_model(bert_save_dir),
			'load_path': '../trained_models/heterozygosity_bin_prepost/version_6/checkpoints/epoch=3-best_val_loss.ckpt'
		},
		{
			'name': 'DNABERT v7',
			'model': make_DNABERT_model(bert_save_dir),
			'load_path': '../trained_models/heterozygosity_bin_prepost/version_7/checkpoints/epoch=1-best_val_loss.ckpt'
		},
		{
			'name': 'DNABERT v9',
			'model': make_DNABERT_model(bert_save_dir),
			'load_path': '../trained_models/heterozygosity_bin_prepost/version_9/checkpoints/epoch=2-best_val_loss.ckpt'
		},
		{
			'name': 'DNABERT v10',
			'model': make_DNABERT_model(bert_save_dir),
			'load_path': '../trained_models/heterozygosity_bin_prepost/version_10/checkpoints/epoch=12-best_val_loss.ckpt'
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

	# Eval non-BERT models
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	split_file = 'split_1_nc6.json'

	data_mod = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		batch_size=256,
		num_workers=3
	)
	data_mod.setup()

	for m in models_to_eval:
		model = STRPrePostClassifier.load_from_checkpoint(
			m['load_path'],
			model=m['model']
		)
		trainer = pl.Trainer(gpus=1)

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
			y_true.int().numpy(), y_pred.numpy()
		)
		res_dict['PR_auc'] = auc(res_dict['PR_recall'], res_dict['PR_prec'])
		
		results.append(res_dict)

	# Eval BERT models
	bert_k = 5

	tokenizer = BertTokenizer.from_pretrained(
		bert_save_dir,
		do_lower_case=False,
		from_tf=False
	)
	data_mod = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		batch_size=256,
		num_workers=3,
		tokenizer=tokenizer,
		k=bert_k
	)
	data_mod.setup()

	for m in bert_models_to_eval:
		model = STRPrePostClassifier.load_from_checkpoint(
			m['load_path'],
			model=m['model'],
			bert=True
		)
		trainer = pl.Trainer(gpus=1)

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
			y_true.int().numpy(), y_pred.numpy()
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

	fig.savefig('../trained_models/heterozygosity_bin_prepost/roc_pr.png',
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
	fig.savefig('../trained_models/heterozygosity_bin_prepost/CMs.png',
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
	table.to_csv('../trained_models/heterozygosity_bin_prepost/table.csv')

	print(table)