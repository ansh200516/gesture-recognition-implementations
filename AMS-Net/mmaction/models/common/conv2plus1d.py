import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils.conv_utils import normalize_tuple

# In mmaction2, CONV_LAYERS is a registry for custom modules.
# TensorFlow doesn't have a direct equivalent for this out of the box.
# We will define the layer directly. A common practice is to use
# tf.keras.utils.get_custom_objects().add_to_scope to register custom layers.
# For this conversion, we will just provide the class.


class Conv2plus1d(layers.Layer):
    """(2+1)d Conv module for R(2+1)d backbone.

    This module is a TensorFlow/Keras implementation of the (2+1)d convolution
    block described in the paper "A Closer Look at Spatiotemporal Convolutions
    for Action Recognition" (https://arxiv.org/pdf/1711.11248.pdf).

    It replaces a 3D convolution with a spatial 2D convolution and a temporal
    1D convolution, which can lead to better performance and fewer parameters.

    Args:
        filters (int): Number of output channels, same as `out_channels` in
            nn.Conv3d.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        strides (int | tuple[int]): Strides of the convolution. Default: 1.
        padding (str | tuple[int]): One of "valid" or "same" (case-insensitive)
            or a tuple of ints. If a tuple of ints is provided, it is
            interpreted as symmetric padding.
        dilation_rate (int | tuple[int]): Dilation rate for the convolution.
            Default: 1.
        groups (int): Number of groups for grouped convolution. This layer
            only supports `groups=1` (the default).
        use_bias (bool): Whether to use a bias term. Default: True.
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer to
            use. If None, `tf.keras.layers.BatchNormalization` is used.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 dilation_rate=1,
                 groups=1,
                 use_bias=True,
                 norm_layer=None,
                 **kwargs):
        super().__init__(**kwargs)

        if groups != 1:
            raise ValueError('groups > 1 is not supported.')

        self.filters = filters
        self.kernel_size = normalize_tuple(kernel_size, 3, 'kernel_size')
        self.strides = normalize_tuple(strides, 3, 'strides')
        self.padding = padding
        self.dilation_rate = normalize_tuple(dilation_rate, 3,
                                             'dilation_rate')
        self.use_bias = use_bias
        self.norm_layer = norm_layer
        if self.norm_layer is None:
            self.norm_layer = layers.BatchNormalization

    def build(self, input_shape):
        """Builds the layer's weights at the time of the first call.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.
        """
        in_channels = input_shape[-1]

        # The middle-plane calculation is based on the paper.
        # M_i = floor((t * d^2 * N_i-1 * N_i) / (d^2 * N_i-1 + t * N_i))
        # where d, t are spatial and temporal kernel sizes, and
        # N_i, N_i-1 are output and input channels, respectively.
        # https://arxiv.org/pdf/1711.11248.pdf
        t_kernel, h_kernel, w_kernel = self.kernel_size
        mid_channels = (t_kernel * h_kernel * w_kernel * in_channels *
                        self.filters)
        mid_channels /= (
            h_kernel * w_kernel * in_channels + t_kernel * self.filters)
        self.mid_channels = int(mid_channels)

        # Spatial convolution
        self.conv_s = layers.Conv3D(
            filters=self.mid_channels,
            kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
            strides=(1, self.strides[1], self.strides[2]),
            padding=self.padding,
            dilation_rate=(1, self.dilation_rate[1], self.dilation_rate[2]),
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name='conv_s')

        self.bn_s = self.norm_layer(name='bn_s')
        self.relu = layers.ReLU()

        # Temporal convolution
        self.conv_t = layers.Conv3D(
            filters=self.filters,
            kernel_size=(self.kernel_size[0], 1, 1),
            strides=(self.strides[0], 1, 1),
            padding=self.padding,
            dilation_rate=(self.dilation_rate[0], 1, 1),
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name='conv_t')

        super().build(input_shape)

    def call(self, x, training=None):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            training (bool): Whether the layer should behave in training mode
                or inference mode.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.conv_s(x)
        x = self.bn_s(x, training=training)
        x = self.relu(x)
        x = self.conv_t(x)
        return x

    def get_config(self):
        """Returns the configuration of the layer."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'norm_layer': self.norm_layer,
        })
        return config
