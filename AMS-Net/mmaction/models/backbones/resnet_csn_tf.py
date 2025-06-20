import tensorflow as tf
from tensorflow.keras import layers

from ..registry import BACKBONES
from .resnet3d_tf import Bottleneck3d, ResNet3d, ConvBnAct


class CSNBottleneck3d(Bottleneck3d):
    """Channel-Separated Bottleneck Block in TensorFlow.

    Args:
        planes (int): Number of channels produced by some norm/conv3d layers.
        bottleneck_mode (str): 'ip' or 'ir'.
        kwargs (dict, optional): Keyword arguments for Bottleneck.
    """

    def __init__(self,
                 planes,
                 *args,
                 bottleneck_mode='ir',
                 **kwargs):
        super().__init__(planes, *args, **kwargs)
        self.bottleneck_mode = bottleneck_mode

        if self.bottleneck_mode not in ['ip', 'ir']:
            raise ValueError(f'Bottleneck mode must be "ip" or "ir",'
                             f'but got {bottleneck_mode}.')

        conv2_kernel_size = self.conv2.conv.kernel_size
        conv2_strides = self.conv2.conv.strides
        conv2_dilation_rate = self.conv2.conv.dilation_rate

        depthwise_conv_block = tf.keras.Sequential([
            layers.Conv3D(
                filters=planes,
                kernel_size=conv2_kernel_size,
                strides=conv2_strides,
                padding='valid',
                dilation_rate=conv2_dilation_rate,
                groups=planes,
                use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ], name='conv2_depthwise_block')

        if self.bottleneck_mode == 'ip':
            ip_conv = layers.Conv3D(
                filters=planes, kernel_size=1, strides=1, use_bias=False, name='conv2_ip')
            self.conv2 = tf.keras.Sequential(
                [ip_conv, depthwise_conv_block], name='conv2')
        else:
            self.conv2 = depthwise_conv_block


@BACKBONES.register_module()
class ResNet3dCSN(ResNet3d):
    """ResNet backbone for CSN in TensorFlow.

    Args:
        depth (int): Depth of ResNetCSN, from {50, 101, 152}.
        bottleneck_mode (str): 'ip' or 'ir'. Default: 'ir'.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    def __init__(self,
                 depth,
                 bottleneck_mode='ir',
                 **kwargs):
        self.bottleneck_mode = bottleneck_mode
        super().__init__(depth, **kwargs)

    def build(self, input_shape):
        self.arch_settings = {
            50: (CSNBottleneck3d, (3, 4, 6, 3)),
            101: (CSNBottleneck3d, (3, 4, 23, 3)),
            152: (CSNBottleneck3d, (3, 8, 36, 3))
        }
        if self.depth not in self.arch_settings:
            raise KeyError(f'invalid depth {self.depth} for ResNet3dCSN')
        self.block, self.stage_blocks = self.arch_settings[self.depth]
        super().build(input_shape)

    def make_res_layer(self, *args, **kwargs):
        """Make a residual layer with bottleneck_mode."""
        return super().make_res_layer(
            *args, **kwargs, bottleneck_mode=self.bottleneck_mode) 