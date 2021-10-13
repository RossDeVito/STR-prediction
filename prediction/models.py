import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection, Precision, Recall, F1
from torchmetrics import ConfusionMatrix


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

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class basic_CNN(nn.Module):
	def __init__(self, input_dim=7, out_channels=32, dropout=.25):
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
