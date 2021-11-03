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
                    in_channels=in_channels, 
                    n_filters=n_filters_pred,
                    kernel_sizes=kernel_sizes, 
                    depth=depth_pred,
                    dropout=dropout, 
                    activation=activation,
                    bottleneck_first=True
                )
            )
        )