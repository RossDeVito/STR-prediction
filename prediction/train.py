import os
import json

import pytorch_lightning as pl

from data_modules import STRDataModule
from models import STRClassifier, basic_CNN


if __name__ == '__main__':
	# Load
	data_dir = os.path.join('..', 'data', 'mecp2_binding', 'samples')
	split_file = 'split_1.json'

	task_log_dir = 'mecp2_logs'

	data = STRDataModule(
		data_dir, 
		split_file, 
		batch_size=128,
		num_workers=4
	)

	model = STRClassifier(basic_CNN(), pos_weight=.001)

	callbacks = [
		# pl.callbacks.LearningRateMonitor('epoch'),
		pl.callbacks.EarlyStopping('val_loss', verbose=True, patience=15),
		pl.callbacks.ModelCheckpoint(
			monitor="val_loss",
			filename='{epoch}-{val_loss:.6f}-{val_F1:.4f}'
		)
	]
	tb_logger = pl.loggers.TensorBoardLogger(os.getcwd(), task_log_dir)
	trainer = pl.Trainer(
		callbacks=callbacks,
		logger=tb_logger,
		gpus=1, 
		log_every_n_steps=1, 
		auto_lr_find=True
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