"""Plot PR and ROC curves for a set of models."""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	save_output = True
	save_desc = 'old_new_ens' # if None will name with timestamp int

	trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1'
	# trained_res_dir = '../trained_models/het_cls_v2/round3/V2_1_T'

	# models_to_plot = [
	# 	# {
	# 	# 	'name': 'version_0',
	# 	# 	'path': 'version_0',
	# 	# 	'which_res': 'all' # Default behavior if not 'which_res' in dict is all.
	# 	# },					   #  Other options are 'best' and 'last'.
	# 	{
	# 		'name': 'version_1',
	# 		'path': 'version_1',
	# 		'which_res': 'best'
	# 	},
	# 	{
	# 		'name': 'version_2',
	# 		'path': 'version_2',
	# 		'which_res': 'best'
	# 	},
	# 	# {
	# 	# 	'name': 'version_3',
	# 	# 	'path': 'version_3',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	{
	# 		'name': 'version_4',
	# 		'path': 'version_4',
	# 		'which_res': 'best'
	# 	},
	# 	# {
	# 	# 	'name': 'version_6',
	# 	# 	'path': 'version_6',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	# {
	# 	# 	'name': 'version_7',
	# 	# 	'path': 'version_7',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	# {
	# 	# 	'name': 'version_8',
	# 	# 	'path': 'version_8',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	# {
	# 	# 	'name': 'version_9',
	# 	# 	'path': 'version_9',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	# {
	# 	# 	'name': 'version_10',
	# 	# 	'path': 'version_10',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	# {
	# 	# 	'name': 'version_11',
	# 	# 	'path': 'version_11',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	# {
	# 	# 	'name': 'version_12',
	# 	# 	'path': 'version_12',
	# 	# 	'which_res': 'best'
	# 	# },
	# 	# Ensembles
	# 	{
	# 		'name': 'ensemble_v1,2,4',
	# 		'path': 'ensembles/ens1_v1v2v4',
	# 		'which_res': 'mean'
	# 	},
	# 	# {
	# 	# 	'name': 'ensemble_v1,2,4',
	# 	# 	'path': 'ensembles/ens1_v1v2v4',
	# 	# 	'which_res': 'agree'
	# 	# },
	# ]
	models_to_plot = [
		{
			'name': 'version_0',
			'path': 'version_0',
			'which_res': 'best'
		},
		# {
		# 	'name': 'version_4',
		# 	'path': 'version_4',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_7',
		# 	'path': 'version_7',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_8',
		# 	'path': 'version_8',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_9',
		# 	'path': 'version_9',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_11',
		# 	'path': 'version_11',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_12',
		# 	'path': 'version_12',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': 'version_13',
		# 	'path': 'version_13',
		# 	'which_res': 'best'
		# },
		# # Ensembles
		# {
		# 	'name': 'ensemble_v0,7,8',
		# 	'path': 'ensembles/ens1_v0v7v8',
		# 	'which_res': 'mean'
		# },
		# {
		# 	'name': 'ensemble_v0,7,8',
		# 	'path': 'ensembles/ens1_v0v7v8',
		# 	'which_res': 'agree'
		# },
		{
			'name': 'ensemble_v0,7,9,12,13',
			'path': 'ensembles/ens1_v0v7v9v12v13',
			'which_res': 'mean'
		},
		# {
		# 	'name': 'ensemble_v0,7,9,12,13',
		# 	'path': 'ensembles/ens1_v0v7v9v12v13',
		# 	'which_res': 'agree'
		# },
		# {
		# 	'name': '5comb_v0',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_0',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': '5comb_v1',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_1',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': '5comb_v2',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_2',
		# 	'which_res': 'best'
		# },
		# {
		# 	'name': '5comb_v3',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_3',
		# 	'which_res': 'best'
		# },
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
		# {
		# 	'name': '5comb_v6',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_6',
		# 	# 'which_res': 'best'
		# },
		# {
		# 	'name': '5comb_v7',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_7',
		# 	# 'which_res': 'best'
		# },
		# {
		# 	'name': '5comb_v10',
		# 	'path': '../V2_1_AC-AG-AT-CT-GT/version_10',
		# 	# 'which_res': 'best'
		# },
		{
			'name': '5comb_ens_v4,6,10',
			'path': '../V2_1_AC-AG-AT-CT-GT/ensembles/ens1_v4v6v10',
			'which_res': 'mean'
		},
	]

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
			elif mod['which_res'] == 'mean':
				res_files = [f for f in res_files if 'mean' in f]
			elif mod['which_res'] == 'agree':
				res_files = [f for f in res_files if 'agree' in f]

		for fname in res_files:
			res_file = os.path.join(res_dir, fname)
			with open(res_file, 'r') as f:
				res = json.load(f)
				if 'best' in fname:
					res['name'] = mod['name'] + ' (best)'
				elif 'last' in fname:
					res['name'] = mod['name'] + ' (last)'
				elif '_results_mean' in fname:
					res['name'] = mod['name'] + ' (mean)'
				elif '_results_vote' in fname:
					res['name'] = mod['name'] + ' (voting)'
				elif '_results_agree' in fname:
					res['name'] = mod['name'] + ' (all agree)'
				else:
					res['name'] = mod['name']
				results.append(res)

	# plot ROC and PR curves
	if save_desc is None:
		time_int = int(datetime.now().timestamp()) # for naming plots
		print(time_int)
	else:
		time_int = save_desc # excuse this variable naming pls
	lw = 2

	fig, (ax_roc, ax_pr_0, ax_pr_1) = plt.subplots(1, 3, sharex=True, sharey=True, 
										subplot_kw=dict(box_aspect=1),
										figsize=(14, 6))
	ax_roc.plot([0, 1], [0, 1], color="black", lw=lw, linestyle=":")

	for res in results:
		ax_roc.plot(res['ROC_fpr'], res['ROC_tpr'], label=res['name'], lw=lw,
					linestyle='--')
		ax_pr_0.plot(res['PR_recall_0'], res['PR_prec_0'], label=res['name'], 
						lw=lw, linestyle='--')
		ax_pr_1.plot(res['PR_recall_1'], res['PR_prec_1'], label=res['name'], 
						lw=lw, linestyle='--')

	fig.suptitle("Binary Heterozygosity Prediction")
	ax_roc.set_title("ROC")
	ax_roc.set_xlabel("False Positive Rate")
	ax_roc.set_ylabel("True Positive Rate")
	ax_roc.legend(loc='lower right')
	ax_pr_0.set_title("PR Curve Non-Het")
	ax_pr_0.set_xlabel("Recall")
	ax_pr_0.set_ylabel("Precision")
	ax_pr_0.legend(loc='lower left')
	ax_pr_1.set_title("PR Curve Het")
	ax_pr_1.set_xlabel("Recall")
	ax_pr_1.set_ylabel("Precision")
	ax_pr_1.legend(loc='lower left')
	fig.tight_layout()

	plt.show()
	if save_output:
		fig.savefig(
			os.path.join(trained_res_dir, 'roc_pr_{}.png'.format(time_int)),
			bbox_inches='tight'
		)
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
	fig.tight_layout()
	plt.show()
	if save_output:
		fig.savefig(
			os.path.join(trained_res_dir, 'CMs_{}.png'.format(time_int)),
			bbox_inches='tight')
	plt.close(fig)

	# Make table
	df = pd.DataFrame(results)

	table = df[['name', 'macro_F1', 'ROC_auc', 'class_precision', 'class_recall']]
	# table['macro_F1'] = [s.item() for s in table.macro_F1]
	# table['ROC_auc'] = [s.item() for s in table.ROC_auc]
	table['class_precision_0'] = [s[0] for s in table.class_precision]
	table['class_precision_1'] = [s[1] for s in table.class_precision]
	table['class_recall_0'] = [s[0] for s in table.class_recall]
	table['class_recall_1'] = [s[1] for s in table.class_recall]
	table = table.drop(columns=['class_precision', 'class_recall'])
	# Get avg rank
	table['Macro_F1_rank'] = table.macro_F1.rank(ascending=False)
	table['ROC_auc_rank'] = table.ROC_auc.rank(ascending=False)
	table['mean_rank'] = (table.Macro_F1_rank + table.ROC_auc_rank) / 2

	if save_output:
		table.to_csv(
			os.path.join(trained_res_dir, 'res_table_{}.csv'.format(time_int)),
		)

	print(table)

	print("Sorted by Macro F1:")
	print(table.sort_values(by='macro_F1', ascending=False))

	print("Sorted by ROC AUC:")
	print(table.sort_values(by='ROC_auc', ascending=False))

	print("Sorted by mean rank:")
	print(table.sort_values(by='mean_rank', ascending=True))


