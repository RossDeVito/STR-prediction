"""Create a voting classifier based on the results jsons from score_models.py
and save results in a similar format to allow for comparison and plotting
(e.g. with plot_curves.py).
"""
import os
import json

import numpy as np
import pandas as pd
import torch
from torchmetrics import MetricCollection
from torchmetrics import ConfusionMatrix, Precision, Recall, F1, PrecisionRecallCurve
from sklearn.metrics import roc_curve, auc, precision_recall_curve


if __name__ == '__main__':
	ensemble_name = 'ens1_v1v2v4'

	# trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1'
	# trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1_T'
	trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1_AC-AG-AT-CT-GT'


	# models_to_plot = [
	# 	{
	# 		'name': 'version_0',
	# 		'path': 'version_0',
	# 		'which_res': 'all' # Default behavior if not 'which_res' in dict is all.
	# 	},					   #  Other options are 'best' and 'last'.
		# {
		# 	'name': 'version_1',
		# 	'path': 'version_1',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_2',
		# 	'path': 'version_2',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_4',
		# 	'path': 'version_4',
		# 	'which_res': 'best'
		# },
	# 	{
	# 		'name': 'version_6',
	# 		'path': 'version_6',
	# 	},
	# 	{
	# 		'name': 'version_7',
	# 		'path': 'version_7',
	# 	},
	# 	{
	# 		'name': 'version_8',
	# 		'path': 'version_8',
	# 	},
	# 	{
	# 		'name': 'version_11',
	# 		'path': 'version_11',
	# 	}
	# ]
	# models_to_plot = [
	# 	{
	# 		'name': 'version_0',
	# 		'path': 'version_0',
	# 		'which_res': 'best'
	# 	},
	# 	# {
	# 	# 	'name': 'version_4',
	# 	# 	'path': 'version_4',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	{
	# 		'name': 'version_7',
	# 		'path': 'version_7',
	# 		'which_res': 'best'
	# 	},
	# 	# {
	# 	# 	'name': 'version_8',
	# 	# 	'path': 'version_8',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	{
	# 		'name': 'version_9',
	# 		'path': 'version_9',
	# 		'which_res': 'best'
	# 	},
	# 	# {
	# 	# 	'name': 'version_11',
	# 	# 	'path': 'version_11',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	{
	# 		'name': 'version_12',
	# 		'path': 'version_12',
	# 		'which_res': 'best'
	# 	},
	# 	{
	# 		'name': 'version_13',
	# 		'path': 'version_13',
	# 		'which_res': 'best'
	# 	}
	# ]

	models_to_plot = [
		{
			'name': '5comb_v4',
			'path': '../V2_1_AC-AG-AT-CT-GT/version_4',
			# 'which_res': 'best'
		},
		# {
		# 	'name': '5comb_v5',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_5',
		# 	# 'which_res': 'best'
		# },
		{
			'name': '5comb_v6',
			'path': '../V2_1_AC-AG-AT-CT-GT/version_6',
			# 'which_res': 'best'
		},
		# {
		# 	'name': '5comb_v7',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_7',
		# 	# 'which_res': 'best'
		# },
		{
			'name': '5comb_v10',
			'path': '../V2_1_AC-AG-AT-CT-GT/version_10',
			# 'which_res': 'best'
		},
	]

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

	# Load results
	results = []

	for mod in models_to_plot:
		res_dir = os.path.join(trained_res_dir, mod['path'], 'results')
		res_files = os.listdir(res_dir)
		if 'which_res' in mod and mod['which_res'] != 'all':
			if mod['which_res'] == 'best':
				res_files = [f for f in res_files if 'best' in f]
			elif mod['which_res'] == 'last':
				res_files = [f for f in res_files if 'last' in f]

		for fname in res_files:
			res_file = os.path.join(res_dir, fname)
			with open(res_file, 'r') as f:
				res = json.load(f)
				if 'best' in fname:
					res['name'] = mod['name'] + ' (best)'
				elif 'last' in fname:
					res['name'] = mod['name'] + ' (last)'
				else:
					res['name'] = mod['name']
				results.append(res)

	# validate ordering of ground truth
	all_y_true = [np.array(r['y_true']) for r in results]
	for i in range(len(all_y_true)):
		if np.any(all_y_true[i] != all_y_true[0]):
			raise ValueError('Ground truth not the same for all models.')

	# Create predictions from votes
	all_preds = np.stack([np.array(r['y_pred']) for r in results])
	y_pred_vote = np.mean((all_preds > .5), axis=0)
	y_pred_mean = np.mean(all_preds, axis=0)
	y_true_all_pos = all_y_true[0]

	# Predictions just where there is agreement
	all_agree_pos = (y_pred_vote == 1.0) | (y_pred_vote == 0.0)
	y_pred_vote_agree = y_pred_mean[all_agree_pos]
	y_true_agree_pos = y_true_all_pos[all_agree_pos]
	
	for y_pred,type in [
			(y_pred_vote,'vote'),(y_pred_mean,'mean'),(y_pred_vote_agree,'agree'),
		]:
		# Subset if just where all agree
		if type == 'agree':
			y_true = y_true_agree_pos
		else:
			y_true = y_true_all_pos

		res_dict = metrics(
			torch.tensor(y_pred), 
			torch.tensor(y_true).int()
		)
		res_dict = {k: v.numpy() for k, v in res_dict.items()}
		res_dict['y_true'] = y_true
		res_dict['y_pred'] = y_pred

		res_dict['num_true_0'] = (y_true == 0).sum().item()
		res_dict['num_true_1'] = (y_true == 1).sum().item()

		res_dict['ROC_fpr'], res_dict['ROC_tpr'], _ = roc_curve(y_true, y_pred)
		res_dict['ROC_auc'] = auc(res_dict['ROC_fpr'], res_dict['ROC_tpr'])

		res_dict['PR_prec_1'], res_dict['PR_recall_1'], _ = precision_recall_curve(
			y_true, y_pred, pos_label=1
		)
		res_dict['PR_auc_1'] = auc(res_dict['PR_recall_1'], res_dict['PR_prec_1'])

		res_dict['PR_prec_0'], res_dict['PR_recall_0'], _ = precision_recall_curve(
			y_true, -y_pred, pos_label=0
		)
		res_dict['PR_auc_0'] = auc(res_dict['PR_recall_0'], res_dict['PR_prec_0'])

		# Save results as lists
		res_dict = {
			k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in res_dict.items() 
		}

		# Save results
		res_save_dir = os.path.join(trained_res_dir, 'ensembles', ensemble_name, 'results')
		if not os.path.exists(res_save_dir):
			os.makedirs(res_save_dir)
		res_save_file = os.path.join(
			res_save_dir, 
			'{}_results_{}.json'.format(ensemble_name, type)
		)
		with open(res_save_file, 'w') as f:
			json.dump(res_dict, f, indent=2)