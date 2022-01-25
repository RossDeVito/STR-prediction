import torch
import torch.nn as nn
import torch.nn.functional as F

import cnn_models



class PrePostModel(nn.Module):
    """ Models which apply a common feature extractor to pre and post
    sequences, then combine embeddings and predict.

    Attributes:
        feature_extractor: Module which extracts features from DNA
            sequence representations.
        predictor: Module which combines embeddings for pre- and pos-
            sequences and produces a prediction logit.
    """
    def __init__(self, feature_extractor, predictor):
        super(PrePostModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.predictor = predictor

    def forward(self, pre_seq, post_seq):
        x_pre = self.feature_extractor(pre_seq)
        x_post = self.feature_extractor(post_seq)
        return self.predictor(x_pre, x_post)

        
class ConcatPredictor(nn.Module):
    """ Predictor which concatenates pre and post embeddings on last 
    axis (position) before passing to a model which predicts logits.

    Attributes:
        embedding_dim: Dimension of embeddings.
        predictor: Module which predicts logits.
    """
    def __init__(self, predictor):
        super(ConcatPredictor, self).__init__()
        self.predictor = predictor

    def forward(self, pre_embed, post_embed):
        x = torch.cat((pre_embed, post_embed), dim=-1)
        return self.predictor(x)


class ConcatPredictorEncoder(nn.Module):
    """ Predictor which concatenates pre and post embeddings on last 
    axis (position) before transposing last two dimensions to pass to
    an encoder. Output of that is globalmaxpooled and passed to a
    classifier.

    Attributes:
        embedding_dim: Dimension of embeddings.
        encoder:
        predictor: Module which predicts logits.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, pre_embed, post_embed):
        x = torch.cat((pre_embed, post_embed), dim=-1)
        x = x.transpose(1,2)
        x = self.encoder(x)
        x = x.max(dim=1).values
        return self.predictor(x)


class InceptionPrePostModel(PrePostModel):
    """ Uses Inception type CNN for both parts of PrePostModel. """
    def __init__(self, in_channels=5, output_dim=1, depth_fe=4, depth_pred=2,
                    kernel_sizes=[9, 19, 39], n_filters_fe=32, 
                    n_filters_pred=32, dropout=0.3,
                    activation='relu'):
        super(InceptionPrePostModel, self).__init__(
            feature_extractor=cnn_models.InceptionBlock(
                in_channels=in_channels, 
                n_filters=n_filters_fe,
                kernel_sizes=kernel_sizes, 
                depth=depth_fe,
                dropout=dropout, 
                activation=activation
            ),
            predictor=ConcatPredictor(
                cnn_models.InceptionTime(
                    in_channels=n_filters_fe*(len(kernel_sizes)+1),
                    n_filters=n_filters_pred,
                    kernel_sizes=kernel_sizes, 
                    depth=depth_pred,
                    dropout=dropout, 
                    activation=activation,
                    bottleneck_first=True
                )
            )
        )


class DenseNet(nn.Module):
	def __init__(self, input_size, layer_sizes, output_size, dropout=0.5):
		super(DenseNet, self).__init__()
		assert len(layer_sizes) > 0

		layers = []

		for i, n_hidden in enumerate(layer_sizes):
			if i == 0:
				layers.append(nn.Linear(input_size, n_hidden))
			else:
				layers.append(nn.Linear(layer_sizes[i-1], n_hidden))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(p=dropout))
		
		layers.append(nn.Linear(layer_sizes[-1], output_size))

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class InceptionPreDimRedPost(PrePostModel):
    """ Uses Inception type CNN feature extractor and a classifier that reduces
    dimentional with attention pooling, then flattens to dense. """
    def __init__(self, n_per_side, reduce_to, in_channels=5, output_dim=1, 
                    depth_fe=4, pool_size=2,
                    kernel_sizes_fe=[9, 19, 39], n_filters_fe=32, 
                    kernel_sizes_pred=[7, 11, 15], n_filters_pred=32, 
                    dropout_cnn=0.2, dropout_dense=0.2, activation='relu',
                    dense_layer_sizes=[128]):
        super(InceptionPreDimRedPost, self).__init__(
            feature_extractor=cnn_models.InceptionBlock(
                in_channels=in_channels, 
                n_filters=n_filters_fe,
                kernel_sizes=kernel_sizes_fe, 
                depth=depth_fe,
                dropout=dropout_cnn, 
                activation=activation
            ),
            predictor=ConcatPredictor(
                nn.Sequential(
                    cnn_models.DimReducingInception(
                        in_channels=n_filters_fe*(len(kernel_sizes_fe)+1),
                        n_filters=n_filters_pred,
                        kernel_sizes=kernel_sizes_pred,
                        in_seq_len=n_per_side * 2,
                        out_seq_len=reduce_to,
                        pool_size=pool_size,
                        dropout=dropout_cnn,
                        activation=activation,
                        bottleneck_first=True
                    ),
                    nn.Flatten(),
                    DenseNet(
                        input_size=reduce_to * (len(kernel_sizes_pred) + 1) * n_filters_pred,
                        layer_sizes=dense_layer_sizes,
                        output_size=output_dim,
                        dropout=dropout_dense,
                    )
                )
            )
        )


if __name__ == '__main__':

    training_params = {
		# Data Module
		'batch_size': 256,
		'min_copy_number': None,
		'max_copy_number': 15,
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
		'pos_weight': None,

		# Callbacks
		'early_stopping_patience': 50,

		# Model params
		'model_type': 'InceptionPreDimRedPost',#'InceptionPrePostModel',
		'depth_fe': 5,
		'n_filters_fe': 32,
		'depth_pred': 2,
		'n_filters_pred': 32,
		'kernel_sizes': [3, 5, 7, 9, 13, 21],
		'activation': 'gelu',
		'dropout': 0.25,

		# for InceptionPreDimRedPost
		'reduce_to': 16,
		'pool_size': 2,
		'kernel_sizes_pred': [5, 7, 9],
		'dropout_dense': 0.2,
		'dense_layer_sizes': [128],
	}

    X1 = torch.rand(8, 6, training_params['window_size'])
    X2 = torch.rand(8, 6, training_params['window_size'])

    model = InceptionPreDimRedPost(n_per_side=64, reduce_to=16, in_channels=6,)
    net = InceptionPreDimRedPost(
			n_per_side=training_params['window_size'],
			reduce_to=training_params['reduce_to'],
			in_channels=6,
			depth_fe=training_params['depth_fe'],
			pool_size=training_params['pool_size'],
			n_filters_fe=training_params['n_filters_fe'],
			kernel_sizes_fe=training_params['kernel_sizes'],
			kernel_sizes_pred=training_params['kernel_sizes_pred'],
			n_filters_pred=training_params['n_filters_pred'],
			activation=training_params['activation'],
			dropout_cnn=training_params['dropout'],
			dropout_dense=training_params['dropout_dense'],
			dense_layer_sizes=training_params['dense_layer_sizes']
		)

    out = model(X1, X2)
    nout = net(X1, X2)