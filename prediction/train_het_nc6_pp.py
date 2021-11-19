import os
import json

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_modules import STRHetPrePostDataModule
from models import STRPrePostClassifier, PrePostModel, InceptionPrePostModel
from models import STRPrePostRegressor

import cnn_models
import enformer_models
import model_utils


if __name__ == '__main__':
	# options
	regression = False

	from_checkpoint = True
	checkpoint_path = 'heterozygosity_logs/full/version_0/checkpoints/epoch=22-last.ckpt'

	# Load
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	split_file = 'split_1.json'

	task_log_dir = 'heterozygosity_logs'
	model_log_dir = 'full'

	data = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		batch_size=32,
		num_workers=3,
		is_binary=(not regression),
	)

	# incep_2
	# net = InceptionPrePostModel(dropout=.4)
	# net = InceptionPrePostModel(
	# 	depth_fe=6,
	# 	n_filters_fe=64,
	# 	depth_pred=3,
	# 	n_filters_pred=64,
	# 	kernel_sizes=[3,7,15,39],
	# 	activation='gelu',
	# 	dropout=.4
	# )
	net = PrePostModel(
		feature_extractor=cnn_models.InceptionBlock(
				in_channels=5, 
				depth=4,
				activation='gelu'
			),
		predictor=enformer_models.EncoderPredictor(
			d_model=128,
			num_layers=2,
			dim_ff=1500,
			n_head=4,
		)
	)
	if regression:
		model = STRPrePostRegressor(
			net, 
			learning_rate=1e-3, 
			reduce_lr_on_plateau=True,
			patience=5
		)
	else:
		model = STRPrePostClassifier(
			net, 
			learning_rate=1e-4, 
			reduce_lr_on_plateau=True,
			patience=15,
			# pos_weight=2.5#1.5#2.0
		)

	print(model_utils.count_params(model))

	callbacks = [
		pl.callbacks.EarlyStopping('val_loss', verbose=True, patience=50),#25),
		pl.callbacks.ModelCheckpoint(
			monitor="val_loss",
			filename='{epoch}-best_val_loss'
		),
		pl.callbacks.ModelCheckpoint(
			save_last=True,
			filename='{epoch}-last'
		)
	]
	tb_logger = pl.loggers.TensorBoardLogger(
		os.path.join(os.getcwd(), task_log_dir), 
		model_log_dir,
		default_hp_metric=False
	)
	if from_checkpoint:
		trainer = pl.Trainer(
			callbacks=callbacks,
			logger=tb_logger,
			gpus=1, 
			log_every_n_steps=1, 
			resume_from_checkpoint=checkpoint_path
		)
	else:
		trainer = pl.Trainer(
			callbacks=callbacks,
			logger=tb_logger,
			gpus=1, 
			log_every_n_steps=1, 
			# max_epochs=3, 
			# limit_train_batches=200,
			# limit_val_batches=200,
			# limit_test_batches=200,
			# auto_lr_find=True
		)

	trainer.tune(model, data)
	print("Learning rate: {}".format(model.learning_rate))
	start_lr = model.learning_rate

	trainer.fit(model, data)

	best_val = trainer.test(
		ckpt_path='best', 
		test_dataloaders=data.test_dataloader()
	)

	print("Best validation Results")
	print(best_val)

	with open(os.path.join(trainer.logger.log_dir, 'best_val.json'), 'w') as fp:
		json.dump(best_val, fp)

	# # val_preds = trainer.predict(
	# # 	ckpt_path='best', 
	# # 	dataloaders=data.val_dataloader()
	# # )

	# # y_hat = np.concatenate([r['y_hat'].cpu().numpy() for r in val_preds])
	# # y_true = np.concatenate([r['y_true'].cpu().numpy() for r in val_preds])
	# # sns.displot({'y_hat': y_hat, 'y_true': y_true})
	# # plt.show()