import tensorflow as tf
from tensorflow.keras import layers
from .ams_2D_module_tf import Competitive_Progressive_Temporal_Module, Temporal_Block


class CPTM_bottleneck(layers.Layer):

    def __init__(
        self,
        channels: int,
        stage=1,
        percent=0.25,
        gamma=1,

        # default settings follow the ones in 2d net
        dim_restore=True,
        relu_out=True,
        short_cut = False,
        conv1d_relu_flag = True,
        **kwargs
    ) -> None:
        super(CPTM_bottleneck, self).__init__(**kwargs)
        self.channels = channels
        self.percent = percent
        self.stage = stage//2
        self.current_channels = int(self.channels*self.percent/self.stage)
        self.gamma = gamma

        self.dim_restore = dim_restore
        self.relu_out = relu_out
        self.short_cut = short_cut
        self.conv1d_relu_flag = conv1d_relu_flag

        #reduce
        self.reduce = layers.Conv3D(self.current_channels, kernel_size=1, strides=1, padding='valid', use_bias=False, data_format='channels_first')
        self.bn_re = layers.BatchNormalization(axis=1)  # syncBN
        
        #CPTM
        self.cptm_bottleneck = \
            Competitive_Progressive_Temporal_Module(
                Temporal_Block, self.current_channels, branchs=3, rate=16, gamma=self.gamma)
        
        #restore
        if self.dim_restore:
            self.restore = layers.Conv3D(self.channels, kernel_size=1, strides=1, padding='valid', use_bias=False, data_format='channels_first')
            self.bn_restore = layers.BatchNormalization(axis=1)  # syncBN
        self.relu = layers.ReLU()

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:

        x = self.reduce(x)
        x = self.bn_re(x, training=training)
        x = self.relu(x)
        if self.short_cut :
            x_dim_reduce = x

        #CPTM
        output = self.cptm_bottleneck(x, training=training)

        if self.dim_restore:
            output = self.restore(output)
            output = self.bn_restore(output, training=training)

            if self.relu_out :
                output = self.relu(output)

        if self.short_cut :
            return output + x_dim_reduce
        else :
            return output 