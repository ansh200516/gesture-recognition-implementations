import tensorflow as tf


class TAM(tf.keras.layers.Layer):
    """Temporal Adaptive Module(TAM) for TANet.

    This module is proposed in `TAM: TEMPORAL ADAPTIVE MODULE FOR VIDEO
    RECOGNITION <https://arxiv.org/pdf/2005.06803>`_

    This is a TensorFlow implementation of the official PyTorch version.
    It assumes 'channels_last' data format.

    Args:
        in_channels (int): Channel num of input features.
        num_segments (int): Number of frame segments.
        alpha (int): `alpha` in the paper and is the ratio of the
            intermediate channel number to the initial channel number in the
            global branch. Default: 2.
        adaptive_kernel_size (int): `K` in the paper and is the size of the
            adaptive kernel size in the global branch. Default: 3.
        beta (int): `beta` in the paper and is set to control the model
            complexity in the local branch. Default: 4.
        conv1d_kernel_size (int): Size of the convolution kernel of Conv1d in
            the local branch. Default: 3.
        adaptive_convolution_stride (int): The first dimension of strides in
            the adaptive convolution of `Temporal Adaptive Aggregation`.
            Default: 1.
        adaptive_convolution_padding (str): The padding in
            the adaptive convolution of `Temporal Adaptive Aggregation`.
            Default: 'same'.
        init_std (float): Std value for initiation of `tf.keras.layers.Dense`.
            Default: 0.001.
    """

    def __init__(self,
                 in_channels,
                 num_segments,
                 alpha=2,
                 adaptive_kernel_size=3,
                 beta=4,
                 conv1d_kernel_size=3,
                 adaptive_convolution_stride=1,
                 adaptive_convolution_padding='same',
                 init_std=0.001,
                 **kwargs):
        super().__init__(**kwargs)

        assert beta > 0 and alpha > 0
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.alpha = alpha
        self.adaptive_kernel_size = adaptive_kernel_size
        self.beta = beta
        self.conv1d_kernel_size = conv1d_kernel_size
        self.adaptive_convolution_stride = adaptive_convolution_stride
        self.adaptive_convolution_padding = adaptive_convolution_padding
        self.init_std = init_std

        # Global branch
        self.G = tf.keras.Sequential([
            tf.keras.layers.Dense(num_segments * self.alpha, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.init_std)),
            tf.keras.layers.BatchNormalization(gamma_initializer='ones'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.adaptive_kernel_size, use_bias=False,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.init_std)),
            tf.keras.layers.Softmax(axis=-1)
        ], name='G')

        # Local branch
        self.L = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                filters=self.in_channels // self.beta,
                kernel_size=self.conv1d_kernel_size,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(gamma_initializer='ones'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(
                filters=self.in_channels,
                kernel_size=1,
                use_bias=False,
                kernel_initializer='he_normal'),
            tf.keras.layers.Activation('sigmoid')
        ], name='L')

    def call(self, x, training=None):
        """Defines the computation performed at every call.

        Args:
            x (tf.Tensor): The input data. Assumes `channels_last` format.
                Shape: `(n, h, w, c)`.
            training (bool): Whether the layer should behave in training mode or
                inference mode.

        Returns:
            tf.Tensor: The output of the module.
        """
        # (n, h, w, c)
        shape = tf.shape(x)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        num_batches = n // self.num_segments
        
        # (num_batches, num_segments, h, w, c)
        x_reshaped = tf.reshape(x, (num_batches, self.num_segments, h, w, c))
        # (num_batches, c, num_segments, h, w)
        x_permuted = tf.transpose(x_reshaped, perm=[0, 4, 1, 2, 3])

        # (num_batches * c, num_segments, 1, 1)
        # GlobalAveragePooling2D on (h, w)
        theta_out = tf.reduce_mean(x_permuted, axis=[3, 4], keepdims=True)
        
        # (num_batches * c, num_segments)
        theta_out_squeezed = tf.squeeze(theta_out, axis=[3, 4])

        # Global branch
        # (num_batches * c, adaptive_kernel_size)
        conv_kernel_raw = self.G(theta_out_squeezed, training=training)
        # (num_batches * c, 1, adaptive_kernel_size, 1)
        conv_kernel = tf.reshape(conv_kernel_raw, (num_batches * c, 1, self.adaptive_kernel_size, 1))

        # Local branch
        # (num_batches * c, num_segments) -> (num_batches, c, num_segments)
        L_in_reshaped = tf.reshape(theta_out_squeezed, (num_batches, c, self.num_segments))
        # (num_batches, num_segments, c)
        L_in_transposed = tf.transpose(L_in_reshaped, perm=[0, 2, 1])
        # self.L expects (batch_size, steps, channels)
        local_activation_raw = self.L(L_in_transposed, training=training) # output is (num_batches, num_segments, c)
        # transpose back to (num_batches, c, num_segments)
        local_activation_transposed = tf.transpose(local_activation_raw, perm=[0, 2, 1])
        # (num_batches, c, num_segments, 1, 1)
        local_activation = tf.reshape(local_activation_transposed, (num_batches, c, self.num_segments, 1, 1))

        # (num_batches, c, num_segments, h, w)
        new_x = x_permuted * local_activation

        # (1, num_batches * c, num_segments, h * w)
        new_x_reshaped = tf.reshape(new_x, (1, num_batches * c, self.num_segments, h * w))
        
        # Perform grouped convolution
        y = tf.nn.conv2d(
            new_x_reshaped,
            filters=conv_kernel,
            strides=[1, self.adaptive_convolution_stride, 1, 1],
            padding='SAME' if self.adaptive_convolution_padding == 'same' else 'VALID',
            data_format='NCHW'
        )

        # (num_batches, c, num_segments, h, w)
        y_reshaped = tf.reshape(y, (num_batches, c, self.num_segments, h, w))
        # (num_batches, num_segments, h, w, c)
        y_permuted = tf.transpose(y_reshaped, perm=[0, 2, 3, 4, 1])
        # (n, h, w, c)
        y_final = tf.reshape(y_permuted, (n, h, w, c))
        
        return y_final 