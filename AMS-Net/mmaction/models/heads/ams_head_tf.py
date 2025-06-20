import tensorflow as tf
from tensorflow.keras import layers


class AvgConsensusTF(layers.Layer):
    """Average consensus module in TensorFlow.

    Args:
        axis (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x, **kwargs):
        """Defines the computation performed at every call."""
        return tf.reduce_mean(x, axis=self.axis, keepdims=True)


class AMSHeadTF(layers.Layer):
    """Class head for AMS in TensorFlow.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):
        super().__init__(name='AMSHeadTF', **kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensusTF(axis=consensus_.get('dim', 1))
        else:
            self.consensus = None

        if self.dropout_ratio > 0:
            self.dropout = layers.Dropout(self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = layers.Dense(
            self.num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=self.init_std),
            bias_initializer='zeros',
            name='fc_cls')

        if self.spatial_type == 'avg':
            self.avg_pool3d = layers.GlobalAveragePooling3D(name='avg_pool3d')
        else:
            self.avg_pool3d = None

        self.avg_pool2d = None
        self.new_cls = None

    def _init_new_cls(self):
        """Initialize the new classification layer for FCN testing."""
        self.new_cls = layers.Conv3D(
            self.num_classes,
            kernel_size=1,
            strides=1,
            padding='valid',
            name='new_cls')

        # Build layer to initialize weights
        dummy_input = tf.zeros((1, 1, 1, 1, self.in_channels))
        self.new_cls(dummy_input)

        fc_weights = self.fc_cls.get_weights()
        conv_kernel = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(fc_weights[0], 0), 0), 0)
        self.new_cls.set_weights([conv_kernel, fc_weights[1]])

    def call(self, x, num_segs=None, fcn_test=False, training=None):
        """Defines the computation performed at every call.

        Args:
            x (tf.Tensor): The input data. Assumes channels-last format.
            num_segs (int | None): Number of segments. Default: None.
            fcn_test (bool): Fully-convolutional test mode. Default: False.
            training (bool): Whether in training mode. Default: None.

        Returns:
            tf.Tensor: The classification scores.
        """
        if fcn_test:
            if self.avg_pool3d:
                x = self.avg_pool3d(x)
                x = tf.reshape(x, (-1, 1, 1, 1, self.in_channels))
            if self.new_cls is None:
                self._init_new_cls()
            cls_score_feat_map = self.new_cls(x)
            return cls_score_feat_map

        if self.avg_pool2d is None and self.spatial_type == 'avg':
            self.avg_pool2d = layers.GlobalAveragePooling2D(
                name='avg_pool2d')

        if num_segs is None:
            # Input: [N, T, H, W, C]
            x = self.avg_pool3d(x)  # Output: [N, C]
        else:
            # Input: [N * num_segs, H, W, C]
            if self.spatial_type == 'avg':
                x = self.avg_pool2d(x)  # Output: [N * num_segs, C]

        if not isinstance(x, tf.Tensor) or len(x.shape) > 2:
            x = layers.Flatten()(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        cls_score = self.fc_cls(x)

        if num_segs is not None and self.consensus is not None:
            cls_score_shape = tf.shape(cls_score)
            cls_score = tf.reshape(cls_score,
                                   (-1, num_segs, cls_score_shape[1]))
            cls_score = self.consensus(cls_score)
            cls_score = tf.squeeze(cls_score, axis=1)

        return cls_score 