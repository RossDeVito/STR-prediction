import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import prepost_models
import cnn_models


class IdentitiyPred(nn.Module):
	def __init__(self):
		super(IdentitiyPred, self).__init__()

	def forward(self, pre_embed, post_embed):
		return pre_embed, post_embed


class EncoderPredictor(prepost_models.ConcatPredictorEncoder):
	def __init__(self, d_model, n_head=8, dim_ff=2048, enc_dropout=0.1,
			num_layers=6):
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_head,
			dim_feedforward=dim_ff,
			dropout=enc_dropout,
			activation="gelu",
			batch_first=True,
		)
		encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		predictor = nn.Linear(d_model, 1)
		super().__init__(encoder, predictor)

if __name__ == '__main__':
	from data_modules import STRHetPrePostDataModule
	from model_utils import count_params

	# Load data
	data_dir = os.path.join('..', 'data', 'mecp2_binding', 'samples_pp')
	split_file = 'split_1.json'

	task_log_dir = 'mecp2_logs'
	model_log_dir = 'incep_4_2_pp'

	data = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		batch_size=4,
		num_workers=3,
		is_binary=True,
	)
	data.setup()
	batch = next(iter(data.train_dataloader()))

	# Create model

	# encoder_layer = nn.TransformerEncoderLayer(
	# 	d_model=128,
	# 	nhead=8,
	# 	batch_first=True,
	# )
	# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

	# predictor = prepost_models.ConcatPredictorEncoder(
	# 	transformer_encoder,
	# 	predictor=nn.Sequential(
	# 		nn.Linear(128, 1)
	# 	)
	# )

	# pp_model = prepost_models.PrePostModel(
	# 	feature_extractor=cnn_models.InceptionBlock(
	# 			in_channels=5, 
	# 			depth=2,
	# 			activation='gelu'
	# 		),
	# 	predictor=predictor
	# )

	pp_model = prepost_models.PrePostModel(
		feature_extractor=cnn_models.InceptionBlock(
				in_channels=5, 
				depth=2,
				activation='gelu'
			),
		predictor=EncoderPredictor(
			d_model=128,
		)
	)
	

	# Run batch
	print(count_params(pp_model))
	res = pp_model(batch['pre_feat_mat'], batch['post_feat_mat'])
	