import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(pre_mat, post_mat, pre_str, post_str, 
		desc=None, label=None, pred=None, cmap='PRGn'):
	max_attr = max(pre_mat.max(), post_mat.max())
	min_attr = min(pre_mat.min(), post_mat.min())

	fig, ax = plt.subplots(1, 2, figsize=(35, 3))
	sns.heatmap(
		data=pre_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance'],
		xticklabels=pre_str,
		ax=ax[0],
		cmap=cmap
	)
	sns.heatmap(
		data=post_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance'],
		xticklabels=post_str,
		ax=ax[1],
		cmap=cmap
	)
	plt.tight_layout()

	if desc is not None:
		if label is not None and pred is not None:
			plt.suptitle('{}    label: {}, pred: {}'.format(desc, label, pred))
		else:
			plt.suptitle('{}'.format(desc))
	elif label is not None and pred is not None:
		plt.suptitle('label: {}, pred: {}'.format(label, pred))

	return fig, ax


def plot_heatmap_in_axs(pre_mat, post_mat, pre_str, post_str, axs,	
		desc=None, label=None, pred=None, cmap='PRGn'):
	max_attr = max(pre_mat.max(), post_mat.max())
	min_attr = min(pre_mat.min(), post_mat.min())

	sns.heatmap(
		data=pre_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance'],
		xticklabels=pre_str,
		ax=axs[0],
		cmap=cmap
	)
	sns.heatmap(
		data=post_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T', 'distance'],
		xticklabels=post_str,
		ax=axs[1],
		cmap=cmap
	)
	plt.tight_layout()

	if desc is not None:
		if label is not None and pred is not None:
			axs[0].set_title('{}    label: {}, pred: {}'.format(desc, label, pred))
		else:
			axs[0].set_title('{}'.format(desc))
	elif label is not None and pred is not None:
		axs[0].set_title('label: {}, pred: {}'.format(label, pred))

	return axs


if __name__ == '__main__':
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

	# Look at correct preds
	correct_df = meta_df[meta_df['label'] == meta_df['pred_bin']]
	non_het_df = correct_df[correct_df['label'] == 0]
	het_df = correct_df[correct_df['label'] == 1]

	# Plot single example
	# ind = 14

	# pre_mat = pre_attrs[ind]
	# post_mat = post_attrs[ind]

	# max_attr = max(pre_mat.max(), post_mat.max())
	# min_attr = min(pre_mat.min(), post_mat.min())

	# fig, ax = plt.subplots(1, 2, figsize=(35, 3))
	# sns.heatmap(
	# 	data=pre_attrs[ind],
	# 	vmin=min_attr,
	# 	vmax=max_attr,
	# 	yticklabels=['A', 'C', 'G', 'T', 'distance'],
	# 	xticklabels=meta_df.loc[ind].pre_string,
	# 	ax=ax[0],
	# 	cmap='pink_r'
	# )
	# sns.heatmap(
	# 	data=post_attrs[ind],
	# 	vmin=min_attr,
	# 	vmax=max_attr,
	# 	yticklabels=['A', 'C', 'G', 'T', 'distance'],
	# 	xticklabels=meta_df.loc[ind].post_string,
	# 	ax=ax[1],
	# 	cmap='pink_r'
	# )
	# plt.tight_layout()
	# plt.show()

	ind = 14
	# fig, ax = plot_heatmap(
	# 	pre_attrs[ind], 
	# 	post_attrs[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	'saliency',
	# 	meta_df.loc[ind].label, 
	# 	meta_df.loc[ind].pred
	# )
	# fig, ax = plot_heatmap(
	# 	pre_attrs_gbp[ind], 
	# 	post_attrs_gbp[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	'gbp',
	# 	meta_df.loc[ind].label, 
	# 	meta_df.loc[ind].pred
	# )
	# fig, ax = plot_heatmap(
	# 	pre_attrs_deconv[ind], 
	# 	post_attrs_deconv[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	'deconv',
	# 	meta_df.loc[ind].label, 
	# 	meta_df.loc[ind].pred
	# )
	# fig, ax = plot_heatmap(
	# 	pre_attrs_gradcam[ind], 
	# 	post_attrs_gradcam[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	'gradcam',
	# 	meta_df.loc[ind].label, 
	# 	meta_df.loc[ind].pred
	# )
	# fig, ax = plot_heatmap(
	# 	pre_attrs_igg[ind],
	# 	post_attrs_igg[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	'ig global',
	# 	meta_df.loc[ind].label,
	# 	meta_df.loc[ind].pred
	# )
	# fig, ax = plot_heatmap(
	# 	pre_attrs_igl[ind],
	# 	post_attrs_igl[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	'ig local',
	# 	meta_df.loc[ind].label,
	# 	meta_df.loc[ind].pred
	# )

	# # examples from email
	# ind = 14#20539#31428
	# # ind = 14110#8151#34003#
	# # fig, ax = plot_heatmap(
	# # 	pre_attrs[ind], 
	# # 	post_attrs[ind],
	# # 	meta_df.loc[ind].pre_string,
	# # 	meta_df.loc[ind].post_string,
	# # 	'saliency',
	# # 	meta_df.loc[ind].label, 
	# # 	meta_df.loc[ind].pred
	# # )
	# fig, axs = plt.subplots(5, 2, figsize=(35, 15))
	# plot_heatmap_in_axs(
	# 	pre_attrs_gbp[ind], 
	# 	post_attrs_gbp[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	axs[0],
	# 	'gbp',
	# 	meta_df.loc[ind].label, 
	# 	meta_df.loc[ind].pred
	# )
	# plot_heatmap_in_axs(
	# 	pre_attrs_deconv[ind], 
	# 	post_attrs_deconv[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	axs[1],
	# 	'deconv',
	# 	meta_df.loc[ind].label, 
	# 	meta_df.loc[ind].pred
	# )
	# plot_heatmap_in_axs(
	# 	pre_attrs_gradcam[ind], 
	# 	post_attrs_gradcam[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	axs[2],
	# 	'gradcam',
	# 	meta_df.loc[ind].label, 
	# 	meta_df.loc[ind].pred
	# )
	# plot_heatmap_in_axs(
	# 	pre_attrs_igg[ind],
	# 	post_attrs_igg[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	axs[3],
	# 	'ig global',
	# 	meta_df.loc[ind].label,
	# 	meta_df.loc[ind].pred
	# )
	# plot_heatmap_in_axs(
	# 	pre_attrs_igl[ind],
	# 	post_attrs_igl[ind],
	# 	meta_df.loc[ind].pre_string,
	# 	meta_df.loc[ind].post_string,
	# 	axs[4],
	# 	'ig local',
	# 	meta_df.loc[ind].label,
	# 	meta_df.loc[ind].pred
	# )
	# plt.show()

	# # plot complements
	# ind = 14
	# ind_c = 15

	# # ind = 2
	# # ind_c = 3

	# pre_mat = pre_attrs[ind]
	# post_mat = post_attrs[ind]
	# pre_mat_c = pre_attrs[ind_c]
	# post_mat_c = post_attrs[ind_c]

	# max_attr = max(pre_mat.max(), post_mat.max(), pre_mat_c.max(), post_mat_c.max())
	# min_attr = min(pre_mat.min(), post_mat.min(), pre_mat_c.min(), post_mat_c.min())

	# fig, ax = plt.subplots(2, 2, figsize=(35, 7))

	# sns.heatmap(
	# 	data=pre_attrs[ind],
	# 	vmin=min_attr,
	# 	vmax=max_attr,
	# 	yticklabels=['A', 'C', 'G', 'T', 'distance'],
	# 	xticklabels=meta_df.loc[ind].pre_string,
	# 	ax=ax[0,0],
	# 	cmap='pink_r'
	# )
	# sns.heatmap(
	# 	data=post_attrs[ind],
	# 	vmin=min_attr,
	# 	vmax=max_attr,
	# 	yticklabels=['A', 'C', 'G', 'T', 'distance'],
	# 	xticklabels=meta_df.loc[ind].post_string,
	# 	ax=ax[0,1],
	# 	cmap='pink_r'
	# )
	# ax[0,0].set_title(
	# 	'Forward    label: {}    pred: {}'.format(
	# 		meta_df.loc[ind].label, meta_df.loc[ind].pred)
	# )

	# sns.heatmap(
	# 	data=np.flip(post_mat_c, 1),
	# 	vmin=min_attr,
	# 	vmax=max_attr,
	# 	yticklabels=['A', 'C', 'G', 'T', 'distance'],
	# 	xticklabels=meta_df.loc[ind_c].post_string[::-1],
	# 	ax=ax[1,0],
	# 	cmap='pink_r'
	# )
	# sns.heatmap(
	# 	data=np.flip(pre_mat_c, 1),
	# 	vmin=min_attr,
	# 	vmax=max_attr,
	# 	yticklabels=['A', 'C', 'G', 'T', 'distance'],
	# 	xticklabels=meta_df.loc[ind_c].pre_string[::-1],
	# 	ax=ax[1,1],
	# 	cmap='pink_r'
	# )
	# ax[1,0].set_title(
	# 	'Complement    label: {}    pred: {}'.format(
	# 		meta_df.loc[ind_c].label, meta_df.loc[ind_c].pred)
	# )
	# plt.tight_layout()
	# plt.show()

	## combined results
	fig, axs = plt.subplots(5, 2, figsize=(35, 15))
	plot_heatmap_in_axs(
		pre_attrs_gbp.mean(axis=0),
		post_attrs_gbp.mean(axis=0),
		meta_df.loc[0].pre_string,
		meta_df.loc[0].post_string,
		axs[0],
		'guided backprop'
	)
	plot_heatmap_in_axs(
		pre_attrs_deconv.mean(axis=0),
		post_attrs_deconv.mean(axis=0),
		meta_df.loc[0].pre_string,
		meta_df.loc[0].post_string,
		axs[1],
		'deconvolution'
	)
	plot_heatmap_in_axs(
		pre_attrs_gradcam.mean(axis=0),
		post_attrs_gradcam.mean(axis=0),
		meta_df.loc[0].pre_string,
		meta_df.loc[0].post_string,
		axs[2],
		'guided gradcam'
	)
	plot_heatmap_in_axs(
		pre_attrs_igg.mean(axis=0),
		post_attrs_igg.mean(axis=0),
		meta_df.loc[0].pre_string,
		meta_df.loc[0].post_string,
		axs[3],
		'IG global'
	)
	plot_heatmap_in_axs(
		pre_attrs_igl.mean(axis=0),
		post_attrs_igl.mean(axis=0),
		meta_df.loc[0].pre_string,
		meta_df.loc[0].post_string,
		axs[4],
		'IG local'
	)
	plt.show()