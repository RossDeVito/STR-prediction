"""Implemented models:
	ResNet():   Matches An's largest model
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import *


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


""" InceptionTime
based on https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py
"""
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019).
# InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

class InceptionModule(nn.Module):
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39],
					use_bottleneck=True, bottleneck_channels=32,
					activation='relu'):
		super().__init__()
		self.in_channels = in_channels
		self.n_filters = n_filters
		self.kernel_sizes = kernel_sizes
		self.use_bottleneck = use_bottleneck
		self.bottleneck_channels = bottleneck_channels
		self.activation_type = activation

		if self.use_bottleneck:
			self.bottleneck = nn.Conv1d(
				self.in_channels, 
				out_channels=self.bottleneck_channels,
				kernel_size=1,
				bias=False
			)
			conv_input_size = self.bottleneck_channels
		else:
			self.bottleneck = nn.Identity()
			conv_input_size = self.in_channels

		self.convs = nn.ModuleList()

		for k in self.kernel_sizes:
			self.convs.append(nn.Conv1d(
				in_channels=conv_input_size,
				out_channels=self.n_filters,
				kernel_size=k,
				stride=1,
				padding=k//2,
				bias=False
			))

		self.convs.append(nn.Sequential(
			nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
			nn.Conv1d(conv_input_size, out_channels=self.n_filters, 
						kernel_size=1, bias=False)
		))

		self.batch_norm = nn.BatchNorm1d(
			num_features=(len(self.kernel_sizes)+1)*n_filters
		)
		self.activation_fn = get_activation_fn(self.activation_type)

	def forward(self, x):
		x = self.bottleneck(x)
		x = torch.cat([l(x) for l in self.convs], dim=1)
		return self.activation_fn(self.batch_norm(x))


class InceptionBlock(nn.Module):
	def __init__(self, in_channels, depth=6, n_filters=32, 
					kernel_sizes=[9, 19, 39], use_residual=True, 
					activation='relu', use_bottleneck=True, 
					dropout=.3, **kwargs):
		super().__init__()
		self.in_channels = in_channels
		self.depth = depth
		self.n_filters = n_filters
		self.kernel_sizes = kernel_sizes
		self.use_residual = use_residual
		self.activation_type = activation
		self.use_bottleneck = use_bottleneck
		self.dropout_p = dropout

		# Create components
		self.inception = nn.ModuleList()
		self.shortcuts = nn.ModuleList()
		self.activations = nn.ModuleList()

		for d in range(depth):
			# Add inception module
			if d == 0:
				self.inception.append(InceptionModule(
					in_channels=self.in_channels, 
					n_filters=self.n_filters,
					use_bottleneck=False,
					**kwargs
				))
			else:
				self.inception.append(InceptionModule(
					in_channels=(len(self.kernel_sizes)+1)*self.n_filters,
					n_filters=self.n_filters,
					use_bottleneck=self.use_bottleneck,
					**kwargs
				))
			
			# Add residual connection every 3 layers
			if self.use_residual and d % 3 == 2:
				if d == 2:
					res_in_dim = self.in_channels
				else:
					res_in_dim = (len(self.kernel_sizes)+1)*self.n_filters

				self.shortcuts.append(nn.Sequential(
					nn.Conv1d(
						in_channels=res_in_dim,
						out_channels=(len(self.kernel_sizes)+1)*self.n_filters,
						kernel_size=1,
						stride=1,
						padding=0,
						bias=False),
					nn.BatchNorm1d((len(self.kernel_sizes)+1)*self.n_filters),
				))

				# Add activations for after residual add
				self.activations.append(
					get_activation_fn(self.activation_type)
				)

	def forward(self, x):
		res = x
		for d in range(self.depth):
			x = self.inception[d](x)
			if self.use_residual and d % 3 == 2:
				x = x + self.shortcuts[d//3](res)
				x = self.activations[d//3](x)
				res = x
		return x


class InceptionTime(nn.Module):
	"""InceptionTime"""
	def __init__(self, in_channels=7, output_dim=1, 
					kernel_sizes=[9, 19, 39], n_filters=32,
					dropout=.3, activation='relu',
					**kwargs):
		super().__init__()
		self.in_channels = in_channels
		self.output_dim = output_dim
		self.kernel_sizes = kernel_sizes
		self.n_filters = n_filters
		self.dropout = dropout
		self.activation = activation
		
		self.inception_block = InceptionBlock(
			in_channels=self.in_channels,
			kernel_sizes=self.kernel_sizes,
			n_filters=self.n_filters,
			dropout=self.dropout,
			activation=self.activation,
			**kwargs
		)
		self.global_pool = nn.AdaptiveMaxPool1d(1)
		self.flatten = nn.Flatten()
		self.fc = nn.Linear(
			(len(self.kernel_sizes)+1) * self.n_filters, 
			self.output_dim
		)

	def forward(self, x):
		x = self.inception_block(x)
		x = self.flatten(self.global_pool(x))
		x = self.fc(x)
		return x


if __name__ == '__main__':
	x = torch.randn(4, 7, 1000)
	incep = InceptionModule(7, n_filters=32, kernel_sizes=[7, 39, 19, 61])
	r_m = incep(x)

	incep_block = InceptionBlock(7, depth=6, n_filters=32)
	r_b = incep_block(x)

	model = InceptionTime(in_channels=7, output_dim=1, kernel_sizes=[7, 19, 39])
	r = model(x)