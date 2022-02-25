import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


if __name__ == '__main__':
	attr_type = ['igg', 'igl'][1]

	# Load data
	save_dir = os.path.join('..', 'attr_data', 'heterozygosity', 
		'incep_v1_4_2_bs256', 'saliency')
	meta_df = pd.read_csv(os.path.join(save_dir, 'meta.csv'))
	
	pre_attrs = np.load(os.path.join(save_dir, 'pre_attrs_saliency.npy'))
	post_attrs = np.load(os.path.join(save_dir, 'post_attrs_saliency.npy'))

	pre_attrs_gbp = np.load(os.path.join(save_dir, 'pre_attrs_gbp.npy'))
	post_attrs_gbp = np.load(os.path.join(save_dir, 'post_attrs_gbp.npy'))

	pre_attrs_deconv = np.load(os.path.join(save_dir, 'pre_attrs_deconv.npy'))
	post_attrs_deconv = np.load(os.path.join(save_dir, 'post_attrs_deconv.npy'))

	pre_attrs_gradcam = np.load(os.path.join(save_dir, 'pre_attrs_gradcam.npy'))
	post_attrs_gradcam = np.load(os.path.join(save_dir, 'post_attrs_gradcam.npy'))

	pre_attrs_igg = np.load(os.path.join(save_dir, 'pre_attrs_ig_global.npy'))
	post_attrs_igg = np.load(os.path.join(save_dir, 'post_attrs_ig_global.npy'))

	pre_attrs_igl = np.load(os.path.join(save_dir, 'pre_attrs_ig_local.npy'))
	post_attrs_igl = np.load(os.path.join(save_dir, 'post_attrs_ig_local.npy'))

	# Add bin label
	meta_df['pred_bin'] = (meta_df['pred'] > .5).astype(int)

	# Look at just correct preds
	correct_meta_df = meta_df[meta_df['label'] == meta_df['pred_bin']]
	non_het_meta_df = correct_meta_df[correct_meta_df['label'] == 0]
	het_meta_df = correct_meta_df[correct_meta_df['label'] == 1]

	# Get attr type for examples where correct
	if attr_type == 'igg':
		correct_attrs_pre = pre_attrs_igg[meta_df['label'] == meta_df['pred_bin']]
		correct_attrs_post = post_attrs_igg[meta_df['label'] == meta_df['pred_bin']]
	elif attr_type == 'igl':
		correct_attrs_pre = pre_attrs_igl[meta_df['label'] == meta_df['pred_bin']]
		correct_attrs_post = post_attrs_igl[meta_df['label'] == meta_df['pred_bin']]
	else:
		raise ValueError('attr_type must be igg or igl')

	# Get STR data from master json 
	## TODO: have get_attrs.py save this data
	master_df = pd.read_json('../data/heterozygosity/labeled_samples_het.json')
	chr_loc_names = []

	for index, row in tqdm(master_df.iterrows(), total=len(master_df)):
		if row['complement'] == True:
			chr_loc_names.append('{} (complement)'.format(row['full_seq_name']))
		elif row['complement'] == False:
			chr_loc_names.append('{}'.format(row['full_seq_name']))
		else:
			raise ValueError('complement not valid (True/False)')

	master_df['chr_loc'] = chr_loc_names
	correct_meta_df = correct_meta_df.merge(
		master_df[['chr_loc', 'str_seq']],
		on='chr_loc'
	)

	# Sort and plot like phi correlation analysis
	window_size = 50
	relevance_by_type = dict()
	for str_start_stop in ['CA', 'AC', 'GT', 'TG']:
		relevance_by_type[str_start_stop] = dict()
		relevance_by_type[str_start_stop]['pre'] = []
		relevance_by_type[str_start_stop]['post'] = []

	cbar_vmax = 0
	cbar_vmin = 0

	for (_, meta_row), pre, post in tqdm(
			zip(correct_meta_df.iterrows(), correct_attrs_pre, correct_attrs_post), 
			total=len(correct_meta_df)):
		pre = pre[:, -window_size:]
		post = post[:, :window_size]

		relevance_by_type[meta_row['str_seq'][:2]]['pre'].append(pre)
		relevance_by_type[meta_row['str_seq'][-2:]]['post'].append(post)

	for k in relevance_by_type.keys():
		relevance_by_type[k]['pre'] = np.mean(relevance_by_type[k]['pre'], axis=0)
		relevance_by_type[k]['post'] = np.mean(relevance_by_type[k]['post'], axis=0)

		if relevance_by_type[k]['pre'].max() > cbar_vmax:
			cbar_vmax = relevance_by_type[k]['pre'].max()
		if relevance_by_type[k]['pre'].min() < cbar_vmin:
			cbar_vmin = relevance_by_type[k]['pre'].min()
		if relevance_by_type[k]['post'].max() > cbar_vmax:
			cbar_vmax = relevance_by_type[k]['post'].max()
		if relevance_by_type[k]['post'].min() < cbar_vmin:
			cbar_vmin = relevance_by_type[k]['post'].min()

	# Create plots
	fig, axs = plt.subplots(
		len(relevance_by_type), 
		2,
		sharex='col',
		figsize=(20,8)
	)
	cbar_ax = fig.add_axes([.92, .3, .03, .4])

	cmap='PRGn'

	for i, (str_type, rels) in enumerate(relevance_by_type.items()):
		if i < (len(relevance_by_type) - 1):
			x_ticks_pre = False
			x_ticks_post = False
		else:
			x_ticks_pre = list(range(-rels['pre'].shape[1], 0))
			x_ticks_post = list(range(1, rels['post'].shape[1] + 1))

		sns.heatmap(
			rels['pre'], 
			ax=axs[i, 0], 
			cbar=True,
			cbar_ax=cbar_ax,
			vmax=cbar_vmax,
			vmin=cbar_vmin,
			center=0.0,
			cmap=cmap, 
			yticklabels=['A', 'C', 'G', 'T'],
			xticklabels=x_ticks_pre
		)
		sns.heatmap(
			rels['post'],
			ax=axs[i, 1], 
			cbar=True, 
			cbar_ax=cbar_ax,
			vmax=cbar_vmax,
			vmin=cbar_vmin,
			center=0.0,
			cmap=cmap, 
			yticklabels=False,
			xticklabels=x_ticks_post
		)
		axs[i, 0].set_ylabel(str_type)
	
	title_dict = {
		'igg': 'Integrated Gradients Global',
		'igl': 'Integrated Gradients Local',
	}

	plt.suptitle(title_dict[attr_type])
	plt.tight_layout(rect=[0, 0, .9, 1])
	plt.savefig('{}_w{}.png'.format(attr_type, window_size))
	plt.show()