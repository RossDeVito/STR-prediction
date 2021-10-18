import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics import ConfusionMatrix, Precision, Recall, F1
from torchmetrics import MeanSquaredError, R2Score


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


""" ResNet """
class L1Block(nn.Module):
	def __init__(self):
		super(L1Block, self).__init__()
		self.conv1 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
		self.bn2 = nn.BatchNorm2d(64)
		self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

	def forward(self, x):
		out = self.layer(x)
		out += x
		out = F.relu(out)
		return out


class L2Block(nn.Module):
	def __init__(self):
		super(L2Block, self).__init__()
		self.conv1 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
		self.conv2 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
		self.bn1 = nn.BatchNorm2d(128)
		self.bn2 = nn.BatchNorm2d(128)
		self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

	def forward(self, x):
		out = self.layer(x)
		out += x
		out = F.relu(out)
		return out


class L3Block(nn.Module):
	def __init__(self):
		super(L3Block, self).__init__()
		self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
		self.conv2 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))
		self.conv3 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))

		self.bn1 = nn.BatchNorm2d(200)
		self.bn2 = nn.BatchNorm2d(200)
		self.bn3 = nn.BatchNorm2d(200)

		self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
								   self.conv2, self.bn2, nn.ReLU(inplace=True),
								   self.conv3, self.bn3)

	def forward(self, x):
		out = self.layer(x)
		out += x
		out = F.relu(out)
		return out


class L4Block(nn.Module):
	def __init__(self):
		super(L4Block, self).__init__()
		self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
		self.bn1 = nn.BatchNorm2d(200)
		self.conv2 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
		self.bn2 = nn.BatchNorm2d(200)
		self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
								   self.conv2, self.bn2)

	def forward(self, x):
		out = self.layer(x)
		out += x
		out = F.relu(out)
		return out


class ResNetStem(nn.Module):
	def __init__(self, in_channels, kernel_size=[5, 5, 3, 3, 1], 
					n_filters=[128, 128, 256, 256, 64]):
		super().__init__()
		self.conv1 = nn.Conv1d(in_channels, n_filters[0], kernel_size[0], 
									padding=kernel_size[0]//2, bias=False)
		self.bn1 = nn.BatchNorm1d(n_filters[0])
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size[1], 
									padding=kernel_size[1]//2, bias=False)
		self.bn2 = nn.BatchNorm1d(n_filters[1])
		self.relu2 = nn.ReLU()
		self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size[2], 
									padding=kernel_size[2]//2, bias=False)
		self.bn3 = nn.BatchNorm1d(n_filters[2])
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv1d(n_filters[2], n_filters[3], kernel_size[3], 
									padding=kernel_size[3]//2, bias=False)
		self.bn4 = nn.BatchNorm1d(n_filters[3])
		self.relu4 = nn.ReLU()
		self.conv5 = nn.Conv1d(n_filters[3], n_filters[4], kernel_size[4], 
									padding=kernel_size[4]//2, bias=False)
		self.bn5 = nn.BatchNorm1d(n_filters[4])
		self.relu5 = nn.ReLU()

	def forward(self, x):
		x = self.relu1(self.bn1(self.conv1(x)))
		x = self.relu2(self.bn2(self.conv2(x)))
		x = self.relu3(self.bn3(self.conv3(x)))
		x = self.relu4(self.bn4(self.conv4(x)))
		x = self.relu5(self.bn5(self.conv5(x)))
		return x


class ResNetUniformBlock(nn.Module):
	def __init__(self, kernel_size, n_filters):
		super().__init__()
		self.conv1 = nn.Conv1d(n_filters, n_filters, kernel_size, 
								padding=kernel_size//2, bias=False)
		self.bn1 = nn.BatchNorm1d(n_filters)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size, 
								padding=kernel_size//2, bias=False)
		self.bn2 = nn.BatchNorm1d(n_filters)
		self.relu2 = nn.ReLU()

	def forward(self, x):
		out = self.relu1(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out)) + x
		return self.relu2(out)


class ResNetNonUniformBlock(nn.Module):
	def __init__(self, n_filters, kernel_size_1=7, kernel_size_2_3=3):
		super().__init__()
		self.conv1 = nn.Conv1d(n_filters, n_filters, kernel_size_1, 
								padding=kernel_size_1//2, bias=False)
		self.bn1 = nn.BatchNorm1d(n_filters)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size_2_3, 
								padding=kernel_size_2_3//2, bias=False)
		self.bn2 = nn.BatchNorm1d(n_filters)
		self.relu2 = nn.ReLU()
		self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size_2_3, 
								padding=kernel_size_2_3//2, bias=False)
		self.bn3 = nn.BatchNorm1d(n_filters)
		self.relu3 = nn.ReLU()

	def forward(self, x):
		out = self.relu1(self.bn1(self.conv1(x)))
		out = self.relu2(self.bn2(self.conv2(x)))
		out = self.bn3(self.conv3(out)) + x
		return self.relu3(out)
		

class ResNet(nn.Module):
	"""ResNet adapted from ChromDragoNN.

	ChromDragoNN was implemented in pytorch and published in Surag Nair, et al, Bioinformatics, 2019.
	The original code can be found in https://github.com/kundajelab/ChromDragoNN

	This ResNet consists of:
		- 2 convolutional layers --> 128 channels, filter size (5,1)
		- 2 convolutional layers --> 256 channels, filter size (3,1)
		- 1 convolutional layers --> 64 channels, filter size (1,1)
		- 2 x L1Block
		- 1 conv layer
		- 2 x L2Block
		- maxpool
		- 1 conv layer
		- 2 x L3Block
		- maxpool
		- 2 x L4Block
		- 1 conv layer
		- maxpool
		- 2 fully connected layers

	L1Block: 2 convolutional layers, 64 channels, filter size (3,1)
	L2Block: 2 convolutional layers, 128 channels, filter size (7,1)
	L3Block: 3 convolutional layers, 200 channels, filter size (7,1), (3,1),(3,1)
	L4Block: 2 convolutional layers, 200 channels, filter size (7,1)
	"""
	def __init__(self, input_len=1000, in_channels=7, output_dim=1, 
					dropout=.3):
		super(ResNet, self).__init__()
		self.input_len = input_len
		self.in_channels = in_channels
		self.dropout = dropout
		self.output_dim = output_dim

		# define model
		self.stem = ResNetStem(self.in_channels)

		# add blocks
		self.L1_block_1 = ResNetUniformBlock(kernel_size=3, n_filters=64)
		self.L1_block_2 = ResNetUniformBlock(kernel_size=3, n_filters=64)
		self.L1_out_conv = nn.Conv1d(64, 128, 3, padding=3//2, bias=False)
		self.L1_out_bn = nn.BatchNorm1d(128)
		self.L1_out_relu = nn.ReLU()

		self.L2_block_1 = ResNetUniformBlock(kernel_size=7, n_filters=128)
		self.L2_block_2 = ResNetUniformBlock(kernel_size=7, n_filters=128)
		self.L2_maxpool = nn.MaxPool1d(3, ceil_mode=True)
		self.L2_out_conv = nn.Conv1d(128, 200, 1, padding=1//2, bias=False)
		self.L2_out_bn = nn.BatchNorm1d(200)
		self.L2_out_relu = nn.ReLU()

		self.L3_block_1 = ResNetNonUniformBlock(n_filters=200)
		self.L3_block_2 = ResNetNonUniformBlock(n_filters=200)
		self.L3_maxpool = nn.MaxPool1d(4, ceil_mode=True)

		self.L4_block_1 = ResNetUniformBlock(kernel_size=7, n_filters=200)
		self.L4_block_2 = ResNetUniformBlock(kernel_size=7, n_filters=200)
		self.L4_out_conv = nn.Conv1d(200, 200, 7, padding=7//2, bias=False)
		self.L4_out_bn = nn.BatchNorm1d(200)
		self.L4_out_relu = nn.ReLU()
		self.L4_maxpool = nn.MaxPool1d(4, ceil_mode=True)

		# Linear output head
		self.flattened_dim = 200 * math.ceil(
			math.ceil(math.ceil(self.input_len / 3) / 4) / 4)
		self.fc1 = nn.Linear(self.flattened_dim, 1000)
		self.bn1 = nn.BatchNorm1d(1000)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(self.dropout)
		self.fc2 = nn.Linear(1000, 1000)
		self.bn2 = nn.BatchNorm1d(1000)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(self.dropout)
		self.fc3 = nn.Linear(1000, self.output_dim)

	def forward(self, x):
		x = self.stem(x)

		x = self.L1_block_2(self.L1_block_1(x))
		x = self.L1_out_relu(self.L1_out_bn(self.L1_out_conv(x)))

		x = self.L2_block_2(self.L2_block_1(x))
		x = self.L2_maxpool(x)
		x = self.L2_out_relu(self.L2_out_bn(self.L2_out_conv(x)))

		x = self.L3_block_2(self.L3_block_1(x))
		x = self.L3_maxpool(x)

		x = self.L4_block_2(self.L4_block_1(x))
		x = self.L4_out_relu(self.L4_out_bn(self.L4_out_conv(x)))
		x = self.L4_maxpool(x)

		x = x.view(-1, self.flattened_dim)
		x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
		x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
		x = self.fc3(x)

		return x