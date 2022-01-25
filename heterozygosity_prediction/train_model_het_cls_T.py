import os
import json

import pytorch_lightning as pl

from data_modules import STRDataModule

from model_utils import count_params
import models
import prepost_models


if __name__ == '__main__':
	__spec__ = None

	# options
	training_params = {
		# Data Module
		'batch_size': 256,
		'min_copy_number': 11.0,
		'max_copy_number': 11.5,
		'incl_STR_feat': True,
		'min_boundary_STR_pos': 6,
		'max_boundary_STR_pos': 6,
		'window_size': 64,
		'bp_dist_units': 1000.0,
		'split_name': 'split_1',

		# Optimizer
		'lr': 1e-4,
		'reduce_lr_on_plateau': True,
		'reduce_lr_factor': 0.5,
		'lr_reduce_patience': 20,
		'pos_weight': .7,

		# Callbacks
		'early_stopping_patience': 50,

		# Model params
		'depth_fe': 3,
		'n_filters_fe': 32,
		'depth_pred': 1,
		'n_filters_pred': 32,
		'kernel_sizes': [3, 5, 7, 9, 13],
		'activation': 'gelu',
		'dropout': 0.2
	}
	num_workers_per_loader = 3

	task_log_dir = 'het_cls_logs'
	model_log_dir = 'V2_1_T'
	num_gpus = 1

	# resuming from checkpoint
	from_checkpoint = False
	checkpoint_path = 'heterozygosity_logs/full/version_0/checkpoints/epoch=22-last.ckpt'

	# Load with DataModule
	data_path = os.path.join(
		'..', 'data', 'heterozygosity', 'sample_data_T_V2_repeat_var.json'
	)
	data_module = STRDataModule(
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

	# Create model
	net = prepost_models.InceptionPrePostModel(
		in_channels=data_module.num_feat_channels(),
		depth_fe=training_params['depth_fe'],
		n_filters_fe=training_params['n_filters_fe'],
		depth_pred=training_params['depth_pred'],
		n_filters_pred=training_params['n_filters_pred'],
		kernel_sizes=training_params['kernel_sizes'],
		activation=training_params['activation'],
		dropout=training_params['dropout']
	)
	model = models.STRPrePostClassifier(
		net,
		learning_rate=training_params['lr'],
		reduce_lr_on_plateau=training_params['reduce_lr_on_plateau'],
		reduce_lr_factor=training_params['reduce_lr_factor'],
		patience=training_params['lr_reduce_patience'],
		pos_weight=training_params['pos_weight']
	)
	training_params['model_n_params'] = count_params(model)
	print(training_params['model_n_params'])

	# Setup training
	callbacks = [
		pl.callbacks.EarlyStopping('val_loss', verbose=True, 
			patience=training_params['early_stopping_patience']),
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
			gpus=num_gpus, 
			log_every_n_steps=1, 
			resume_from_checkpoint=checkpoint_path
		)
	else:
		trainer = pl.Trainer(
			callbacks=callbacks,
			logger=tb_logger,
			gpus=num_gpus, 
			log_every_n_steps=1, 
			# max_epochs=3, 
			# limit_train_batches=10,
			# limit_val_batches=200,
			# limit_test_batches=200,
			# auto_lr_find=True
		)

	# Train model
	trainer.fit(model, data_module)
	
	# Get performance on test set
	best_val = trainer.test(
		ckpt_path='best', 
		dataloaders=data_module.test_dataloader()
	)
	print("Best validation Results")
	print(best_val)

	# Save results and parameters
	with open(os.path.join(trainer.logger.log_dir, 'best_val.json'), 'w') as fp:
		json.dump(best_val, fp)

	with open(os.path.join(trainer.logger.log_dir, 'train_params.json'), 'w') as fp:
		json.dump(training_params, fp)
