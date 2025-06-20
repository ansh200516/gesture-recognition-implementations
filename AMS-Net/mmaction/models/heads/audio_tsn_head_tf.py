import tensorflow as tf
from tensorflow.keras import layers

from .base_tf import BaseHeadTF

# Note: registry functionality is not implemented in this snippet.
# from ..registry import HEADS
# @HEADS.register_module()

class AudioTSNHeadTF(BaseHeadTF):
    """Classification head for TSN on audio in TensorFlow.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.spatial_type == 'avg':
            self.avg_pool = layers.GlobalAveragePooling2D(data_format='channels_first')
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = layers.Dropout(self.dropout_ratio)
        else:
            self.dropout = None
        
        self.fc_cls = layers.Dense(
            self.num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.init_std)
        )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # Weight initialization is handled in the layer constructor in TF Keras.
        # This method is kept for API consistency.
        pass

    def call(self, x, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (tf.Tensor): The input data.
                (N, in_channels, H, W) for 'channels_first' data format.

        Returns:
            tf.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, h, w]
        if self.avg_pool:
            x = self.avg_pool(x)
        # [N, in_channels]
        if self.dropout is not None:
            x = self.dropout(x, **kwargs)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score 