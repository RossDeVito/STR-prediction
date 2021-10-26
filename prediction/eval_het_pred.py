"""Load trained models and compare performance."""

import os

import pytorch_lightning as pl

from data_modules import STRDataModuleZonzeroClass
from models import STRClassifier, ResNet


if __name__ == '__main__':
	models_to_eval = [
		{
			'name': 'ResNet 1, lr=.001',
			'model': ResNet(),
			'load_path': '../trained_models/heterozygosity_bin/resnet_1_lr001/checkpoints/epoch=8-best_val_loss.ckpt'
		},
		{   'name': 'ResNet 1, lr=.00025',
			'model': ResNet(),
			'load_path': '../trained_models/heterozygosity_bin/resnet_1_lr00025/checkpoints/epoch=8-best_val_loss.ckpt'
		},
	]

	# Load data splits
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples')
	split_file = 'split_1.json'

	data_mod = STRDataModuleZonzeroClass(
		data_dir, 
		split_file, 
		batch_size=256,
		num_workers=3
	)
	data_mod.setup()

	# Make predictions and evaluate with trained each model
	results = []

	for m in models_to_eval:
		model = STRClassifier.load_from_checkpoint(
			m['load_path'],
			model=m['model']
		)
		trainer = pl.Trainer(weights_summary=None)

		preds = trainer.predict(
			model=model, 
			dataloaders=data_mod.val_dataloader()
		)

		break