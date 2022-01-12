import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics import ConfusionMatrix, Precision, Recall, F1
from torchmetrics import MeanSquaredError, R2Score

from cnn_models import *
from prepost_models import *


class STRPrePostClassifier(pl.LightningModule):
	def __init__(self, model, pos_weight=None, learning_rate=1e-3,
			reduce_lr_on_plateau=False, reduce_lr_factor=0.1, patience=10,
			bert=False, fe_learning_rate=1e-5):
		super().__init__()
		self.model = model
		self.pos_weight = pos_weight
		self.learning_rate = learning_rate
		self.reduce_lr_on_plateau = reduce_lr_on_plateau
		self.reduce_lr_factor = reduce_lr_factor
		self.patience = patience
		self.bert = bert
		self.fe_learning_rate = fe_learning_rate

		self.save_hyperparameters('learning_rate', 'pos_weight', 
			'reduce_lr_on_plateau', 'reduce_lr_factor', 'patience')

		# Metrics
		metrics = MetricCollection([
			Precision(num_classes=2, average='macro', multiclass=True),
			Recall(num_classes=2, average='macro', multiclass=True),
			F1(num_classes=2, average='macro', multiclass=True),
		])
		self.train_metrics = metrics.clone(prefix='train_')
		self.val_metrics = metrics.clone(prefix='val_')
		self.test_metrics = metrics.clone(prefix='test_')

	def forward(self, x_pre, x_post):
		return torch.sigmoid(self.model(x_pre, x_post))

	def shared_step(self, batch):
		if self.bert:
			x_pre = {
				'input_ids': batch['pre_input_ids'],
				'token_type_ids': batch['pre_token_type_ids'],
				'attention_mask': batch['pre_attention_mask']
			}
			x_post = {
				'input_ids': batch['post_input_ids'],
				'token_type_ids': batch['post_token_type_ids'],
				'attention_mask': batch['post_attention_mask']
			}
		else:
			x_pre = batch['pre_STR_feats']
			x_post = batch['post_STR_feats']
		y = batch['label']
		logits = self.model(x_pre, x_post)

		if self.pos_weight is not None:
			weight = torch.tensor([self.pos_weight], device=self.device)
		else:
			weight = self.pos_weight

		loss = F.binary_cross_entropy_with_logits(
			logits, y.unsqueeze(1).float(), weight=weight
		)
		return loss, logits, y

	def training_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.train_metrics(torch.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, on_epoch=True)
		self.log("train_loss", loss, on_step=True, on_epoch=True,
					prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.val_metrics(torch.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("val_loss", loss, prog_bar=True)

	def test_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.test_metrics(torch.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("test_loss", loss, prog_bar=True)

	def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
		if self.bert:
			x_pre = {
				'input_ids': batch['pre_input_ids'],
				'token_type_ids': batch['pre_token_type_ids'],
				'attention_mask': batch['pre_attention_mask']
			}
			x_post = {
				'input_ids': batch['post_input_ids'],
				'token_type_ids': batch['post_token_type_ids'],
				'attention_mask': batch['post_attention_mask']
			}
			return {
				'y_hat': self(x_pre, x_post).flatten(), 
				'y_true': batch['label']
			}
		else:
			return {
				'y_hat': self(batch['pre_STR_feats'], batch['post_STR_feats']).flatten(), 
				'y_true': batch['label']
			}

	def configure_optimizers(self):
		if self.bert:
			params = [
				{'params': self.model.feature_extractor.parameters(), 
				 'lr': self.fe_learning_rate},
				{'params': self.model.predictor.parameters()}
			]
		else:
			params = self.parameters()

		if self.reduce_lr_on_plateau:
			optimizer = torch.optim.Adam(params, lr=self.learning_rate)
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer, 
				factor=self.reduce_lr_factor, 
				patience=self.patience,
				verbose=True
			)
			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': scheduler,
					'monitor': 'val_loss',
				}
			}
		else:
			return torch.optim.Adam(params, lr=self.learning_rate)


class basic_CNN(nn.Module):
	def __init__(self, seq_len, n_channels=7, output_dim=1):
		super().__init__()
		self.conv1 = nn.Conv1d(input_dim, out_channels, 15)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(p=dropout)
		self.conv2 = nn.Conv1d(out_channels, out_channels, 15)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(p=dropout)
		self.conv3 = nn.Conv1d(out_channels, out_channels, 15)
		self.relu3 = nn.ReLU()
		self.dropout3 = nn.Dropout(p=dropout)
		self.classifier = nn.Linear(out_channels, 1)

	def forward(self, x):
		x = self.dropout1(self.relu1(self.conv1(x)))
		x = self.dropout2(self.relu2(self.conv2(x)))
		x = self.dropout3(self.relu3(self.conv3(x)))
		x = torch.max(x, axis=2).values
		return self.classifier(x)