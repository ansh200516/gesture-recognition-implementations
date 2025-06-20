import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import HeNormal, Constant

from ...utils import get_root_logger
from ..registry import BACKBONES


class BasicBlock(layers.Layer):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (tf.keras.Model | None): Downsample layer. Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(
                planes,
                kernel_size=3,
                strides=stride,
                padding='same',
                use_bias=False,
                dilation_rate=dilation,
                kernel_initializer=HeNormal()),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(
                planes,
                kernel_size=3,
                strides=1,
                padding='same',
                use_bias=False,
                dilation_rate=1,
                kernel_initializer=HeNormal()),
            layers.BatchNormalization()
        ])

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.with_cp = with_cp

    def call(self, x, training=None):
        """Defines the computation performed at every call."""
        def _inner_forward(x):
            identity = x

            out = self.conv1(x, training=training)
            out = self.conv2(out, training=training)

            if self.downsample is not None:
                identity = self.downsample(x, training=training)

            out = layers.add([out, identity])
            out = self.relu(out)
            return out

        if self.with_cp:
            return tf.recompute_grad(_inner_forward)(x)
        return _inner_forward(x)


class Bottleneck(layers.Layer):
    """Bottleneck block for ResNet."""

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        if style == 'pytorch':
            conv1_stride = 1
            conv2_stride = stride
        else:
            conv1_stride = stride
            conv2_stride = 1

        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(
                planes,
                kernel_size=1,
                strides=conv1_stride,
                use_bias=False,
                kernel_initializer=HeNormal()),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(
                planes,
                kernel_size=3,
                strides=conv2_stride,
                padding='same',
                dilation_rate=dilation,
                use_bias=False,
                kernel_initializer=HeNormal()),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.conv3 = tf.keras.Sequential([
            layers.Conv2D(
                planes * self.expansion,
                kernel_size=1,
                use_bias=False,
                kernel_initializer=HeNormal()),
            layers.BatchNormalization()
        ])

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.with_cp = with_cp

    def call(self, x, training=None):
        """Defines the computation performed at every call."""
        def _inner_forward(x):
            identity = x

            out = self.conv1(x, training=training)
            out = self.conv2(out, training=training)
            out = self.conv3(out, training=training)

            if self.downsample is not None:
                identity = self.downsample(x, training=training)

            out = layers.add([out, identity])
            return out

        if self.with_cp:
            out = tf.recompute_grad(_inner_forward)(x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


@BACKBONES.register_module()
class ResNet(Model):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        dilations (Sequence[int]): Dilation of each stage.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: ``pytorch``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
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
                 pretrained=None,
                 in_channels=3,
                 num_stages=4,
                 out_indices=(3, ),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = self._make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            # setattr is used to dynamically assign layers to the model
            setattr(self, f'layer{i + 1}', res_layer)
            self.res_layers.append(res_layer)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(
                64,
                kernel_size=7,
                strides=2,
                padding='same',
                use_bias=False,
                kernel_initializer=HeNormal(),
                input_shape=(None, None, self.in_channels)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

    def _make_res_layer(self,
                        block,
                        inplanes,
                        planes,
                        blocks,
                        stride=1,
                        dilation=1,
                        style='pytorch',
                        with_cp=False):
        """Build residual layer for ResNet."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                layers.Conv2D(
                    planes * block.expansion,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False,
                    kernel_initializer=HeNormal()),
                layers.BatchNormalization()
            ])

        layer_list = []
        layer_list.append(
            block(
                inplanes,
                planes,
                stride,
                dilation,
                downsample,
                style=style,
                with_cp=with_cp))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layer_list.append(
                block(
                    inplanes,
                    planes,
                    1,
                    dilation,
                    style=style,
                    with_cp=with_cp))

        return tf.keras.Sequential(layer_list)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # In TensorFlow, layers are initialized when they are built.
        # Keras' default initializers are often sufficient.
        # Custom initialization can be done here if needed.
        # The original code supports loading from a PyTorch checkpoint,
        # which is not directly compatible. A conversion script would be
        # needed for that.
        for m in self.modules:
            if isinstance(m, layers.Conv2D):
                m.kernel_initializer = HeNormal()
            elif isinstance(m, layers.BatchNormalization):
                m.gamma_initializer = Constant(1)
                m.beta_initializer = Constant(0)

    def call(self, x, training=None):
        """Defines the computation performed at every call."""
        if self.norm_eval:
            training = False
        
        x = self.conv1(x, training=training)
        x = self.maxpool(x)
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x, training=training)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.trainable = False

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.trainable = False

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules:
                if isinstance(m, layers.BatchNormalization):
                    m.eval() 