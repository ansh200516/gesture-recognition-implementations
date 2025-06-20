import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

from ..builder import HEADS
from .base_tf import BaseHead


@HEADS.register_module()
class SlowFastHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = layers.Dropout(self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = layers.Dense(
            self.num_classes,
            kernel_initializer=RandomNormal(stddev=self.init_std))

        if self.spatial_type == 'avg':
            self.avg_pool = layers.GlobalAveragePooling3D()
        else:
            self.avg_pool = None

    def call(self, x, training=None):
        """Defines the computation performed at every call.

        Args:
            x (tuple[tf.Tensor]): The input data, a tuple containing
                the fast and slow pathways.

        Returns:
            tf.Tensor: The classification scores for input samples.
        """
        # ([N, T, H, W, C_fast], [N, T, H, W, C_slow])
        x_fast, x_slow = x

        # ([N, C_fast], [N, C_slow])
        if self.avg_pool:
            x_fast = self.avg_pool(x_fast)
            x_slow = self.avg_pool(x_slow)

        # [N, C_fast + C_slow]
        x = tf.concat((x_slow, x_fast), axis=1)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # [N, num_classes]
        cls_score = self.fc_cls(x)

        return cls_score 