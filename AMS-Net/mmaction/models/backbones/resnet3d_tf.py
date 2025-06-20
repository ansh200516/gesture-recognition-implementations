import tensorflow as tf
from tensorflow.keras import layers

from ..registry import BACKBONES
from ...utils.misc import _ntuple


def _get_padding_shape(padding_torch, kernel_size, stride):
    """Calculate padding shape for tf.keras.layers.ZeroPadding3D"""
    padding_tf = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    for i in range(3):
        if padding_torch[i] > 0:
            padding_tf[i + 1] = (padding_torch[i], padding_torch[i])
    return padding_tf


class ConvBnAct(layers.Layer):
    """A helper layer for Conv3D -> BN -> Activation."""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 dilation_rate=(1, 1, 1),
                 use_bias=False,
                 activation='relu',
                 name=None):
        super().__init__(name=name)
        self.conv = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias)
        self.bn = layers.BatchNormalization()
        if activation:
            self.act = layers.Activation(activation)
        else:
            self.act = None

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.act:
            x = self.act(x)
        return x


class BasicBlock3d(layers.Layer):
    """BasicBlock 3d block for ResNet3D.

    Args:
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (tf.keras.layers.Layer | None): Downsample layer.
            Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 1

    def __init__(self,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert style in ['pytorch', 'caffe']

        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.with_cp = with_cp
        self.downsample = downsample

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.padding1 = layers.ZeroPadding3D(
            padding=conv1_padding, name='padding1')
        self.conv1 = ConvBnAct(
            planes,
            conv1_kernel_size,
            strides=(self.conv1_stride_t, self.conv1_stride_s,
                     self.conv1_stride_s),
            dilation_rate=(1, dilation, dilation),
            use_bias=False,
            name='conv1')

        self.padding2 = layers.ZeroPadding3D(
            padding=conv2_padding, name='padding2')
        self.conv2 = ConvBnAct(
            planes,
            conv2_kernel_size,
            strides=(self.conv2_stride_t, self.conv2_stride_s,
                     self.conv2_stride_s),
            activation=None,
            use_bias=False,
            name='conv2')

        self.relu = layers.ReLU(name='relu')

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        identity = x

        out = self.padding1(x)
        out = self.conv1(out, training=training)
        out = self.padding2(out)
        out = self.conv2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out = layers.add([out, identity])
        out = self.relu(out)

        return out


class Bottleneck3d(layers.Layer):
    """Bottleneck 3d block for ResNet3D.

    Args:
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (tf.keras.layers.Layer | None): Downsample layer.
            Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 4

    def __init__(self,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 inflate_style='3x1x1',
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']

        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.with_cp = with_cp
        self.downsample = downsample

        if self.style == 'pytorch':
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.padding1 = layers.ZeroPadding3D(
            padding=conv1_padding, name='padding1')
        self.conv1 = ConvBnAct(
            planes,
            conv1_kernel_size,
            strides=(self.conv1_stride_t, self.conv1_stride_s,
                     self.conv1_stride_s),
            use_bias=False,
            name='conv1')

        self.padding2 = layers.ZeroPadding3D(
            padding=conv2_padding, name='padding2')
        self.conv2 = ConvBnAct(
            planes,
            conv2_kernel_size,
            strides=(self.conv2_stride_t, self.conv2_stride_s,
                     self.conv2_stride_s),
            dilation_rate=(1, dilation, dilation),
            use_bias=False,
            name='conv2')

        self.conv3 = ConvBnAct(
            planes * self.expansion, (1, 1, 1),
            use_bias=False,
            activation=None,
            name='conv3')

        self.relu = layers.ReLU(name='relu')

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        identity = x

        out = self.padding1(x)
        out = self.conv1(out, training=training)
        out = self.padding2(out)
        out = self.conv2(out, training=training)
        out = self.conv3(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out = layers.add([out, identity])
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNet3d(tf.keras.Model):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Default: ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: ``(5, 7, 7)``.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 2.
        with_pool2 (bool): Whether to use pool2. Default: True.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (1, 1, 1, 1).
        inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(5, 7, 7),
                 conv1_stride_t=2,
                 pool1_stride_t=2,
                 with_pool2=True,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 **kwargs):
        super().__init__(**kwargs)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_t = pool1_stride_t
        self.with_pool2 = with_pool2
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.inflate_style = inflate_style
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                name=f'layer{i + 1}')
            self.inplanes = planes * self.block.expansion
            self.res_layers.append(res_layer)

    def make_res_layer(self, block, inplanes, planes, blocks,
                       spatial_stride, temporal_stride, dilation, style,
                       inflate, inflate_style, with_cp, name):
        """Build residual layer for ResNet3D."""
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvBnAct(
                planes * block.expansion,
                kernel_size=(1, 1, 1),
                strides=(temporal_stride, spatial_stride, spatial_stride),
                use_bias=False,
                activation=None,
                name='downsample')

        layer_list = []
        layer_list.append(
            block(
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                with_cp=with_cp,
                name='block1'))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layer_list.append(
                block(
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    with_cp=with_cp,
                    name=f'block{i+1}'))

        return tf.keras.Sequential(layer_list, name=name)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        padding = tuple([(k - 1) // 2 for k in self.conv1_kernel])
        self.padding1 = layers.ZeroPadding3D(padding=padding, name='pad_stem')
        self.conv1 = ConvBnAct(
            self.base_channels,
            kernel_size=self.conv1_kernel,
            strides=(self.conv1_stride_t, 2, 2),
            use_bias=False,
            name='conv1_stem')

        self.maxpool = layers.MaxPool3D(
            pool_size=(1, 3, 3),
            strides=(self.pool1_stride_t, 2, 2),
            padding='same',
            name='maxpool_stem')

        self.pool2 = layers.MaxPool3D(
            pool_size=(2, 1, 1), strides=(2, 1, 1), name='pool2')

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.trainable = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.trainable = False

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        if self.norm_eval:
            for layer in self.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False

        x = self.padding1(x)
        x = self.conv1(x, training=training)
        x = self.maxpool(x)
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x, training=training)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, layers.BatchNormalization):
                    m.trainable = False


@BACKBONES.register_module()
class ResNet3dLayer(tf.keras.layers.Layer):
    """ResNet 3d Layer.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stage (int): The index of Resnet stage. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        spatial_stride (int): The 1st res block's spatial stride. Default 2.
        temporal_stride (int): The 1st res block's temporal stride. Default 1.
        dilation (int): The dilation. Default: 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        all_frozen (bool): Frozen all modules in the layer. Default: False.
        inflate (int): Inflate Dims of each block. Default: 1.
        inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
    """

    def __init__(self,
                 depth,
                 stage=3,
                 base_channels=64,
                 spatial_stride=2,
                 temporal_stride=1,
                 dilation=1,
                 style='pytorch',
                 all_frozen=False,
                 inflate=1,
                 inflate_style='3x1x1',
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 **kwargs):

        super().__init__(**kwargs)
        self.arch_settings = ResNet3d.arch_settings
        assert depth in self.arch_settings

        self.depth = depth
        self.stage = stage
        assert stage >= 0 and stage <= 3
        self.base_channels = base_channels
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.all_frozen = all_frozen
        self.stage_inflation = inflate
        self.inflate_style = inflate_style
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        block, stage_blocks = self.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        self.res_layer = ResNet3d.make_res_layer(
            self,
            block,
            inplanes,
            planes,
            stage_block,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            dilation=dilation,
            style=self.style,
            inflate=self.stage_inflation,
            inflate_style=self.inflate_style,
            with_cp=with_cp,
            name=f'layer{stage + 1}')

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.all_frozen:
            self.res_layer.trainable = False

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        if self.norm_eval:
            for layer in self.layers:
                if hasattr(layer, 'layers'):
                    for sub_layer in layer.layers:
                        if isinstance(sub_layer, layers.BatchNormalization):
                            sub_layer.trainable = False
        self._freeze_stages()
        out = self.res_layer(x, training=training)
        return out


try:
    from mmdet.models.builder import SHARED_HEADS as MMDET_SHARED_HEADS
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:
    MMDET_SHARED_HEADS.register_module()(ResNet3dLayer)
    MMDET_BACKBONES.register_module()(ResNet3d) 