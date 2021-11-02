import os
import json

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_modules import STRDataModule, STRDataModuleZonzeroClass
from models import STRClassifier, ResNet, InceptionTime


if __name__ == '__main__':
	# torch.multiprocessing.set_sharing_strategy('file_system')

	# options
	from_checkpoint = False
	checkpoint_path = 'heterozygosity_logs/resnet_1/version_4/checkpoints/epoch=27-last.ckpt'

	# Load
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples3')
	split_file = 'split_1.json'

	task_log_dir = 'heterozygosity_logs'
	model_log_dir = 'resnet_1_lr001'

	data = STRDataModuleZonzeroClass(
		data_dir, 
		split_file, 
		batch_size=256,
		num_workers=3
	)

	model = STRClassifier(ResNet(), learning_rate=1e-3)

	callbacks = [
		# pl.callbacks.LearningRateMonitor('epoch'),
		pl.callbacks.EarlyStopping('val_loss', verbose=True, patience=25),
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
			# limit_train_batches=20,
			# limit_val_batches=20,
			# limit_test_batches=20,
			# auto_lr_find=True
		)

	trainer.tune(model, data)
	trainer.fit(model, data)

	best_val = trainer.test(
		ckpt_path='best', 
		test_dataloaders=data.val_dataloader()
	)

	print("Best validation Results")
	print(best_val)

	with open(os.path.join(trainer.logger.log_dir, 'best_val.json'), 'w') as fp:
		json.dump(best_val, fp)