import os
import json

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_modules import STRHetPrePostDataModule
from models import STRPrePostClassifier, PrePostModel, InceptionPrePostModel


if __name__ == '__main__':

	# options
	from_checkpoint = False
	checkpoint_path = 'heterozygosity_logs/resnet_1/version_4/checkpoints/epoch=27-last.ckpt'

	# Load
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	split_file = 'split_1_nc6.json'

	task_log_dir = 'heterozygosity_logs'
	model_log_dir = 'incep_4_2_pp_nc6'

	data = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		batch_size=64,
		num_workers=3
	)

	# incep_2
	net = InceptionPrePostModel()
	model = STRPrePostClassifier(
		net, 
		learning_rate=1e-3, 
		reduce_lr_on_plateau=True,
		patience=10
	)

	callbacks = [
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
	print("Learning rate: {}".format(model.learning_rate))
	start_lr = model.learning_rate

	trainer.fit(model, data)

	best_val = trainer.test(
		ckpt_path='best', 
		test_dataloaders=data.val_dataloader()
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