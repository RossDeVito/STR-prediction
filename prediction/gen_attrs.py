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
		batch_size=32,
		num_workers=3
	)
	data_mod.setup()
	data_loader = data_mod.test_dataloader()

	# Setup attr methods
	saliency = attr.Saliency(model)
	gbp = attr.GuidedBackprop(model)
	deconv = attr.Deconvolution(model)
	gradcam = attr.GuidedGradCam(model,
		model.model.predictor.predictor.inception_block.inception[1])
	ig_global = attr.IntegratedGradients(model, multiply_by_inputs=True)
	ig_local = attr.IntegratedGradients(model, multiply_by_inputs=False)
	deep_lift = attr.DeepLift(model, multiply_by_inputs=True)
	deep_list_shap = attr.DeepLiftShap(model, multiply_by_inputs=True)
	grad_shap = attr.GradientShap(model, multiply_by_inputs=True)

	# Data to save
	pre_strings = []
	post_strings = []

	# pre_attrs = []
	# post_attrs = []
	# pre_attrs_gbp = []
	# post_attrs_gbp = []
	# pre_attrs_deconv = []
	# post_attrs_deconv = []
	# pre_attrs_gradcam = []
	# post_attrs_gradcam = []
	# pre_attrs_igg = []
	# post_attrs_igg = []
	# pre_attrs_igl = []
	# post_attrs_igl = []

	pre_attrs_dl = []
	post_attrs_dl = []
	pre_attrs_dls = []
	post_attrs_dls = []
	pre_attrs_gs = []
	post_attrs_gs = []

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
		preds = model(*batch_feats).flatten().tolist()
	
		# grads = saliency.attribute(batch_feats, target=None)
		# gbp_attrs = gbp.attribute(batch_feats, target=None)
		# deconv_attrs = deconv.attribute(batch_feats, target=None)
		# gradcam_attrs = gradcam.attribute(batch_feats, target=None)
		# igg_attrs = ig_global.attribute(batch_feats, target=None, 
		# 	internal_batch_size=64)
		# igl_attrs = ig_local.attribute(batch_feats, target=None,
		# 	internal_batch_size=64)

		# dl_attrs = deep_lift.attribute(batch_feats, target=None)
		dls_attrs = deep_list_shap.attribute(batch_feats, target=None)
		gs_attrs = grad_shap.attribute(batch_feats, target=None)

		# Store data and results
		pre_strings.extend(batch['pre_seq_string'])
		post_strings.extend(batch['post_seq_string'])

		# pre_attrs.append(grads[0].cpu().numpy())
		# post_attrs.append(grads[1].cpu().numpy())
		# pre_attrs_gbp.append(gbp_attrs[0].cpu().numpy())
		# post_attrs_gbp.append(gbp_attrs[1].cpu().numpy())
		# pre_attrs_deconv.append(deconv_attrs[0].detach().cpu().numpy())
		# post_attrs_deconv.append(deconv_attrs[1].detach().cpu().numpy())
		# pre_attrs_gradcam.append(gradcam_attrs[0].detach().cpu().numpy())
		# post_attrs_gradcam.append(gradcam_attrs[1].detach().cpu().numpy())
		# pre_attrs_igg.append(igg_attrs[0].detach().cpu().numpy())
		# post_attrs_igg.append(igg_attrs[1].detach().cpu().numpy())
		# pre_attrs_igl.append(igl_attrs[0].detach().cpu().numpy())
		# post_attrs_igl.append(igl_attrs[1].detach().cpu().numpy())
		
		# pre_attrs_dl.append(dl_attrs[0].detach().cpu().numpy())
		# post_attrs_dl.append(dl_attrs[1].detach().cpu().numpy())
		pre_attrs_dls.append(dls_attrs[0].detach().cpu().numpy())
		post_attrs_dls.append(dls_attrs[1].detach().cpu().numpy())
		pre_attrs_gs.append(gs_attrs[0].detach().cpu().numpy())
		post_attrs_gs.append(gs_attrs[1].detach().cpu().numpy())

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

	# pre_attrs = np.concatenate(pre_attrs, axis=0)
	# np.save(os.path.join(save_dir, 'pre_attrs_saliency.npy'), pre_attrs)
	# pre_attrs_gbp = np.concatenate(pre_attrs_gbp, axis=0)
	# np.save(os.path.join(save_dir, 'pre_attrs_gbp.npy'), pre_attrs_gbp)
	# pre_attrs_deconv = np.concatenate(pre_attrs_deconv, axis=0)
	# np.save(os.path.join(save_dir, 'pre_attrs_deconv.npy'), pre_attrs_deconv)
	# pre_attrs_gradcam = np.concatenate(pre_attrs_gradcam, axis=0)
	# np.save(os.path.join(save_dir, 'pre_attrs_gradcam.npy'), pre_attrs_gradcam)
	# pre_attrs_igg = np.concatenate(pre_attrs_igg, axis=0)
	# np.save(os.path.join(save_dir, 'pre_attrs_ig_global.npy'), pre_attrs_igg)
	# pre_attrs_igl = np.concatenate(pre_attrs_igl, axis=0)
	# np.save(os.path.join(save_dir, 'pre_attrs_ig_local.npy'), pre_attrs_igl)

	# pre_attrs_dl = np.concatenate(pre_attrs_dl, axis=0)
	# np.save(os.path.join(save_dir, 'pre_attrs_dl.npy'), pre_attrs_dl)
	pre_attrs_dls = np.concatenate(pre_attrs_dls, axis=0)
	np.save(os.path.join(save_dir, 'pre_attrs_dls.npy'), pre_attrs_dls)
	pre_attrs_gs = np.concatenate(pre_attrs_gs, axis=0)
	np.save(os.path.join(save_dir, 'pre_attrs_gs.npy'), pre_attrs_gs)

	# post_attrs = np.concatenate(post_attrs, axis=0)
	# np.save(os.path.join(save_dir, 'post_attrs_saliency.npy'), post_attrs)
	# post_attrs_gbp = np.concatenate(post_attrs_gbp, axis=0)
	# np.save(os.path.join(save_dir, 'post_attrs_gbp.npy'), post_attrs_gbp)
	# post_attrs_deconv = np.concatenate(post_attrs_deconv, axis=0)
	# np.save(os.path.join(save_dir, 'post_attrs_deconv.npy'), post_attrs_deconv)
	# post_attrs_gradcam = np.concatenate(post_attrs_gradcam, axis=0)
	# np.save(os.path.join(save_dir, 'post_attrs_gradcam.npy'), post_attrs_gradcam)
	# post_attrs_igg = np.concatenate(post_attrs_igg, axis=0)
	# np.save(os.path.join(save_dir, 'post_attrs_ig_global.npy'), post_attrs_igg)
	# post_attrs_igl = np.concatenate(post_attrs_igl, axis=0)
	# np.save(os.path.join(save_dir, 'post_attrs_ig_local.npy'), post_attrs_igl)

	# post_attrs_dl = np.concatenate(post_attrs_dl, axis=0)
	# np.save(os.path.join(save_dir, 'post_attrs_dl.npy'), post_attrs_dl)
	post_attrs_dls = np.concatenate(post_attrs_dls, axis=0)
	np.save(os.path.join(save_dir, 'post_attrs_dls.npy'), post_attrs_dls)
	post_attrs_gs = np.concatenate(post_attrs_gs, axis=0)
	np.save(os.path.join(save_dir, 'post_attrs_gs.npy'), post_attrs_gs)
