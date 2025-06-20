import tensorflow as tf
from tensorflow.keras import layers, Model


class C3D(Model):
    """C3D backbone.
    This is a TensorFlow/Keras implementation of the C3D model from the paper:
    "Learning Spatiotemporal Features with 3D Convolutional Networks"
    (https://arxiv.org/abs/1412.0767)
    Note: The original PyTorch implementation used a registry system
    (@BACKBONES.register_module()) from mmcv to build models from configs.
    This has been removed as it is specific to the OpenMMLab framework.
    The original implementation also supported loading pretrained weights from a
    PyTorch checkpoint file. This TensorFlow version can load weights from a
    TensorFlow checkpoint (`model.load_weights('path/to/weights')`), but a
    direct conversion from a PyTorch checkpoint is required and is not
    handled here.
    Args:
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation of fc layers. Default: 0.005.
    """

    def __init__(self,
                 dropout_ratio=0.5,
                 init_std=0.005,
                 **kwargs):
        super(C3D, self).__init__(**kwargs)
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        # Kernel initializer for convolutional layers
        conv_kernel_initializer = tf.keras.initializers.HeNormal()
        # Kernel initializer for dense layers
        dense_kernel_initializer = tf.keras.initializers.RandomNormal(stddev=self.init_std)

        self.conv1a = layers.Conv3D(
            filters=64, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv1a')
        self.pool1 = layers.MaxPool3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')

        self.conv2a = layers.Conv3D(
            filters=128, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv2a')
        self.pool2 = layers.MaxPool3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')

        self.conv3a = layers.Conv3D(
            filters=256, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv3a')
        self.conv3b = layers.Conv3D(
            filters=256, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv3b')
        self.pool3 = layers.MaxPool3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')

        self.conv4a = layers.Conv3D(
            filters=512, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv4a')
        self.conv4b = layers.Conv3D(
            filters=512, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv4b')
        self.pool4 = layers.MaxPool3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')

        self.conv5a = layers.Conv3D(
            filters=512, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv5a')
        self.conv5b = layers.Conv3D(
            filters=512, kernel_size=3, activation='relu', padding='same',
            kernel_initializer=conv_kernel_initializer, name='conv5b')

        self.zeropad5 = layers.ZeroPadding3D(padding=(0, 1, 1), name='zeropad5')
        self.pool5 = layers.MaxPool3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')

        self.flatten = layers.Flatten(name='flatten')
        self.fc6 = layers.Dense(
            4096, activation='relu',
            kernel_initializer=dense_kernel_initializer, name='fc6')
        self.dropout = layers.Dropout(rate=self.dropout_ratio)
        self.fc7 = layers.Dense(
            4096, activation='relu',
            kernel_initializer=dense_kernel_initializer, name='fc7')

    def call(self, x, training=False):
        """Defines the computation performed at every call.
        Args:
            x (tf.Tensor): The input data.
                The tensor should have a shape of (num_batches, 16, 112, 112, 3)
                for default C3D, which is (N, D, H, W, C).
            training (bool): Official flag for keras model.
        Returns:
            tf.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.zeropad5(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc6(x)
        x = self.dropout(x, training=training)
        x = self.fc7(x)

        return x
