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


class STRClassifier(pl.LightningModule):
	def __init__(self, model, pos_weight=None, learning_rate=1e-3):
		super().__init__()
		self.model = model
		self.pos_weight = pos_weight
		self.learning_rate = learning_rate

		# Metrics
		metrics = MetricCollection([
			Precision(num_classes=2, average='macro', multiclass=True),
			Recall(num_classes=2, average='macro', multiclass=True),
			F1(num_classes=2, average='macro', multiclass=True),
		])
		self.train_metrics = metrics.clone(prefix='train_')
		self.val_metrics = metrics.clone(prefix='val_')
		self.test_metrics = metrics.clone(prefix='test_')

	def forward(self, x):
		return F.sigmoid(self.model(x))

	def shared_step(self, batch):
		x = batch['feat_mat']
		y = batch['label']
		logits = self.model(x)

		if self.pos_weight is not None:
			weight = torch.tensor([self.pos_weight], device=self.device)
		else:
			weight = self.pos_weight

		loss = F.binary_cross_entropy_with_logits(logits, y.unsqueeze(1).float(),
													weight=weight)
		return loss, logits, y

	def training_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.train_metrics(F.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, on_epoch=True)
		self.log("train_loss", loss, on_step=True, on_epoch=True,
					prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.val_metrics(F.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("val_loss", loss, prog_bar=True)

	def test_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.test_metrics(F.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("test_loss", loss, prog_bar=True)

	def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
		return {
			'y_hat': self(batch['feat_mat']).flatten(), 
			'y_true': batch['label']
		}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class STRPrePostClassifier(pl.LightningModule):
	def __init__(self, model, pos_weight=None, learning_rate=1e-3):
		super().__init__()
		self.model = model
		self.pos_weight = pos_weight
		self.learning_rate = learning_rate

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
		return F.sigmoid(self.model(x_pre, x_post))

	def shared_step(self, batch):
		x_pre = batch['pre_feat_mat']
		x_post = batch['post_feat_mat']
		y = batch['label']
		logits = self.model(x_pre, x_post)

		if self.pos_weight is not None:
			weight = torch.tensor([self.pos_weight], device=self.device)
		else:
			weight = self.pos_weight

		loss = F.binary_cross_entropy_with_logits(logits, y.unsqueeze(1).float(),
													weight=weight)
		return loss, logits, y

	def training_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.train_metrics(F.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, on_epoch=True)
		self.log("train_loss", loss, on_step=True, on_epoch=True,
					prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.val_metrics(F.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("val_loss", loss, prog_bar=True)

	def test_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.test_metrics(F.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("test_loss", loss, prog_bar=True)

	def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
		return {
			'y_hat': self(batch['pre_feat_mat'], batch['post_feat_mat']).flatten(), 
			'y_true': batch['label']
		}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class STRRegressor(pl.LightningModule):
	def __init__(self, model, learning_rate=1e-3):
		super().__init__()
		self.model = model
		self.learning_rate = learning_rate

		# Metrics
		metrics = MetricCollection({
			'R2Score': R2Score(),
			'VarWeightedR2Score': R2Score(multioutput='variance_weighted'),
		})
		self.train_metrics = metrics.clone(prefix='train_')
		self.val_metrics = metrics.clone(prefix='val_')
		self.test_metrics = metrics.clone(prefix='test_')

	def forward(self, x):
		return self.model(x)

	def shared_step(self, batch):
		x = batch['feat_mat']
		y = batch['label']
		y_hat = self.model(x)

		loss = F.mse_loss(y_hat.flatten(), y.float())
		return loss, y_hat, y

	def training_step(self, batch, batch_idx):
		loss, y_hat, y = self.shared_step(batch)
		metrics_dict = self.train_metrics(y_hat.flatten(), y)
		self.log_dict(metrics_dict, on_epoch=True)
		self.log("train_loss", loss, on_step=True, on_epoch=True,
					prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, y_hat, y = self.shared_step(batch)
		metrics_dict = self.val_metrics(y_hat.flatten(), y)
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("val_loss", loss, prog_bar=True)

	def test_step(self, batch, batch_idx):
		loss, y_hat, y = self.shared_step(batch)
		metrics_dict = self.test_metrics(y_hat.flatten(), y)
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("test_loss", loss, prog_bar=True)

	def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
		return {
			'y_hat': self(batch['feat_mat']).flatten(), 
			'y_true': batch['label']
		}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


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