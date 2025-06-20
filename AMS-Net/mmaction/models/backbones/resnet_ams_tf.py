import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from .ams_2D_module_tf import CPTM_Bottleneck, MultiScale_Temporal


class BasicBlock(layers.Layer):
    """TensorFlow implementation of BasicBlock for ResNet.

    Args:
        in_planes (int): Number of channels for the input.
        planes (int): Number of channels for the output.
        stride (int): Stride for the first convolution. Default: 1.
        dilation (int): Dilation for the first convolution. Default: 1.
        downsample (tf.keras.layers.Layer | None): Downsample layer. Default: None.
        data_format (str): 'channels_first' or 'channels_last'. Default: 'channels_first'.
    """
    expansion = 1

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 data_format='channels_first',
                 **kwargs):
        super().__init__(**kwargs)
        bn_axis = 1 if data_format == 'channels_first' else -1

        # In TensorFlow Keras, 'padding' can be 'valid' or 'same'.
        # To replicate PyTorch's explicit padding, we use ZeroPadding2D.
        self.conv1_padding = layers.ZeroPadding2D(
            padding=dilation, data_format=data_format, name='conv1_pad')
        self.conv1 = layers.Conv2D(
            planes,
            kernel_size=3,
            strides=stride,
            padding='valid',
            use_bias=False,
            dilation_rate=dilation,
            data_format=data_format,
            name='conv1')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn1')
        self.relu1 = layers.ReLU(name='relu1')

        self.conv2_padding = layers.ZeroPadding2D(
            padding=1, data_format=data_format, name='conv2_pad')
        self.conv2 = layers.Conv2D(
            planes,
            kernel_size=3,
            strides=1,
            padding='valid',
            use_bias=False,
            dilation_rate=1,
            data_format=data_format,
            name='conv2')
        self.bn2 = layers.BatchNormalization(axis=bn_axis, name='bn2')

        self.relu_out = layers.ReLU(name='relu_out')
        self.downsample = downsample

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        identity = x

        out = self.conv1_padding(x)
        out = self.conv1(out)
        out = self.bn1(out, training=training)
        out = self.relu1(out)

        out = self.conv2_padding(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out = layers.add([out, identity])
        out = self.relu_out(out)

        return out


class Bottleneck(layers.Layer):
    """TensorFlow implementation of Bottleneck for ResNet.

    Args:
        in_planes (int): Number of channels for the input.
        planes (int): Number of channels for the intermediate convolutions.
        stride (int): Stride for the second convolution. Default: 1.
        dilation (int): Dilation for the second convolution. Default: 1.
        downsample (tf.keras.layers.Layer | None): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. Controls stride placement. Default: 'pytorch'.
        with_cp (bool): Use gradient checkpointing. Default: False.
        data_format (str): 'channels_first' or 'channels_last'. Default: 'channels_first'.
    """
    expansion = 4

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 data_format='channels_first',
                 **kwargs):
        super().__init__(**kwargs)
        bn_axis = 1 if data_format == 'channels_first' else -1

        if style == 'pytorch':
            conv1_stride = 1
            conv2_stride = stride
        else:  # caffe
            conv1_stride = stride
            conv2_stride = 1

        self.conv1 = layers.Conv2D(
            planes,
            kernel_size=1,
            strides=conv1_stride,
            use_bias=False,
            data_format=data_format,
            name='conv1')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn1')
        self.relu1 = layers.ReLU(name='relu1')

        self.conv2_padding = layers.ZeroPadding2D(
            padding=dilation, data_format=data_format, name='conv2_pad')
        self.conv2 = layers.Conv2D(
            planes,
            kernel_size=3,
            strides=conv2_stride,
            padding='valid',
            dilation_rate=dilation,
            use_bias=False,
            data_format=data_format,
            name='conv2')
        self.bn2 = layers.BatchNormalization(axis=bn_axis, name='bn2')
        self.relu2 = layers.ReLU(name='relu2')

        self.conv3 = layers.Conv2D(
            planes * self.expansion,
            kernel_size=1,
            strides=1,
            use_bias=False,
            data_format=data_format,
            name='conv3')
        self.bn3 = layers.BatchNormalization(axis=bn_axis, name='bn3')

        self.relu_out = layers.ReLU(name='relu_out')
        self.downsample = downsample
        self.with_cp = with_cp

    def _inner_forward(self, x, training=False):
        """Forward wrapper for utilizing checkpoint."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)

        out = self.conv2_padding(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out = layers.add([out, identity])
        return out

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        if self.with_cp and training:
            out = tf.recompute_grad(self._inner_forward)(x, training=training)
        else:
            out = self._inner_forward(x, training=training)

        out = self.relu_out(out)
        return out


def make_res_layer(block,
                   in_planes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   data_format='channels_first',
                   name=None):
    """Build a residual layer for ResNet."""
    downsample = None
    bn_axis = 1 if data_format == 'channels_first' else -1

    if stride != 1 or in_planes != planes * block.expansion:
        downsample = Sequential(
            [
                layers.Conv2D(
                    planes * block.expansion,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False,
                    data_format=data_format,
                    name='0'),
                layers.BatchNormalization(axis=bn_axis, name='1')
            ],
            name='downsample')

    res_layers = [
        block(
            in_planes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            data_format=data_format,
            name='0')
    ]
    in_planes = planes * block.expansion
    for i in range(1, blocks):
        res_layers.append(
            block(
                in_planes,
                planes,
                1,
                dilation,
                style=style,
                with_cp=with_cp,
                data_format=data_format,
                name=str(i)))

    return Sequential(res_layers, name=name)


class ResNetAMS(Model):
    """TensorFlow implementation of ResNet-AMS backbone.

    Note:
        - Pretrained weight loading is not implemented. Keras models can
          load weights via `model.load_weights()`, but this requires a
          TensorFlow-compatible checkpoint file.
        - `frozen_stages`, `norm_eval`, `partial_bn` are training-time
          configurations. In TensorFlow, this is typically handled by setting
          `layer.trainable = False` in the training script before compilation.
          Helper methods are provided to facilitate this.
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 out_indices=(3, ),
                 CompetitiveFusion=True,
                 temporal_block_indices=(2, ),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 with_cp=False,
                 num_segments=8,
                 gamma=1,
                 data_format='channels_first',
                 **kwargs):
        super().__init__(**kwargs)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError("data_format must be 'channels_first' or "
                             "'channels_last'")

        self.depth = depth
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.strides = strides
        self.dilations = dilations
        self.style = style
        self.with_cp = with_cp
        self.CompetitiveFusion = CompetitiveFusion
        self.gamma = gamma
        self.temporal_block_indices = temporal_block_indices
        self.num_segments = num_segments
        self.data_format = data_format

        bn_axis = 1 if self.data_format == 'channels_first' else -1

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        # Stem Layer
        self.conv1_pad = layers.ZeroPadding2D(
            padding=3, data_format=self.data_format, name='conv1_pad')
        self.conv1 = layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            padding='valid',
            use_bias=False,
            data_format=self.data_format,
            name='conv1')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn1')
        self.relu = layers.ReLU(name='relu')
        self.maxpool_pad = layers.ZeroPadding2D(
            padding=1, data_format=self.data_format, name='maxpool_pad')
        self.maxpool = layers.MaxPool2D(
            pool_size=3, strides=2, padding='valid', data_format=self.data_format, name='max_pool')

        # Temporal Attention Blocks
        if self.CompetitiveFusion:
            self.multiScale_res2 = CPTM_Bottleneck(
                256,
                gamma=self.gamma,
                segments=self.num_segments,
                data_format=self.data_format,
                name='multiScale_res2')
            self.multiScale_res4 = CPTM_Bottleneck(
                1024,
                gamma=self.gamma,
                segments=self.num_segments,
                data_format=self.data_format,
                name='multiScale_res4')
        else:
            self.multiScale_res2 = MultiScale_Temporal(
                256,
                segments=self.num_segments,
                data_format=self.data_format,
                name='multiScale_res2')
            self.multiScale_res4 = MultiScale_Temporal(
                1024,
                segments=self.num_segments,
                data_format=self.data_format,
                name='multiScale_res4')

        # Residual Layers
        self.res_layers = []
        in_planes = 64
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                in_planes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=self.with_cp,
                data_format=self.data_format,
                name=f'layer{i + 1}')
            in_planes = planes * self.block.expansion
            self.res_layers.append(res_layer)
            # Dynamically add layer as attribute for easy access
            setattr(self, f'layer{i+1}', res_layer)

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        x = self.conv1_pad(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool_pad(x)
        x = self.maxpool(x)

        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x, training=training)
            if i in self.temporal_block_indices:
                if i == 0:
                    x = self.multiScale_res2(x, training=training)
                elif i == 2:
                    x = self.multiScale_res4(x, training=training)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def _freeze_stages(self, frozen_stages):
        """Freeze stages of the model."""
        if frozen_stages >= 0:
            self.bn1.trainable = False
            self.conv1.trainable = False

        for i in range(1, frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.trainable = False

        if frozen_stages >= 2:
            self.multiScale_res2.trainable = False

    def _partial_bn(self):
        """Set all BatchNorm layers to eval mode except the first one."""
        print("Freezing BatchNorm layers except the first one.")
        count_bn = 0
        for m in self.submodules:
            if isinstance(m, layers.BatchNormalization):
                count_bn += 1
                if count_bn >= 2:
                    m.trainable = False 