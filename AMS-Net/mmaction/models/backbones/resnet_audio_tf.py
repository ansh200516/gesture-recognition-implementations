import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class ConvAudio(layers.Layer):
    """A custom convolution layer for audio processing.

    This layer mimics the behavior of a factorized convolution found in some
    audio processing networks. It consists of two parallel 2D convolutions.
    The outputs of these convolutions can either be concatenated or summed.

    Args:
        filters (int): The number of output filters for each convolution.
        kernel_size (int | tuple[int]): The kernel size for the convolutions.
        strides (int | tuple[int]): The strides for the convolutions.
        padding (int | tuple[int]): The padding to be applied.
        dilation_rate (int | tuple[int]): The dilation rate for the convolutions.
        op (str): The operation to perform on the outputs of the two
            convolutions. Can be 'concat' or 'sum'. Default: 'concat'.
        data_format (str): 'channels_first' or 'channels_last'.
            Default: 'channels_first'.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 dilation_rate=1,
                 op='concat',
                 data_format='channels_first',
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.op = op
        self.data_format = data_format

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding_layer = layers.ZeroPadding2D(
            padding=padding, data_format=data_format)

        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)

        self.conv1 = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding='valid',
            use_bias=False,
            dilation_rate=dilation_rate,
            data_format=data_format,
            name='conv_a')
        self.conv2 = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding='valid',
            use_bias=False,
            dilation_rate=dilation_rate,
            data_format=data_format,
            name='conv_b')

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        x = self.padding_layer(x)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        if self.op == 'concat':
            axis = 1 if self.data_format == 'channels_first' else -1
            return layers.concatenate([out1, out2], axis=axis)
        if self.op == 'sum':
            return layers.add([out1, out2])
        # This path should not be reached with valid op.
        return None 


class Bottleneck2dAudioTF(layers.Layer):
    """TensorFlow implementation of the Bottleneck2dAudio block.

    Args:
        in_planes (int): Number of channels for the input.
        planes (int): Number of channels for the intermediate convolutions.
        stride (int): Stride for the second convolution. Default: 1.
        dilation (int): Dilation for the second convolution. Default: 1.
        downsample (tf.keras.layers.Layer | None): Downsample layer.
            Default: None.
        factorize (bool): Whether to use factorized convolution.
            Default: True.
        with_cp (bool): Use gradient checkpointing. Default: False.
        data_format (str): 'channels_first' or 'channels_last'.
            Default: 'channels_first'.
    """
    expansion = 4

    def __init__(self,
                 in_planes,
                 planes,
                 stride=2,
                 dilation=1,
                 downsample=None,
                 factorize=True,
                 with_cp=False,
                 data_format='channels_first',
                 **kwargs):
        super().__init__(**kwargs)
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.factorize = factorize
        self.with_cp = with_cp
        self.data_format = data_format
        bn_axis = 1 if data_format == 'channels_first' else -1

        self.conv1 = layers.Conv2D(
            planes,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name='conv1')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn1')
        self.relu1 = layers.ReLU(name='relu1')

        conv2_padding = (dilation, dilation)
        if self.factorize:
            self.conv2 = ConvAudio(
                planes,
                kernel_size=3,
                strides=stride,
                padding=conv2_padding,
                dilation_rate=dilation,
                op='concat',
                data_format=data_format,
                name='conv2_audio')
        else:
            self.conv2_padding = layers.ZeroPadding2D(
                padding=conv2_padding,
                data_format=data_format,
                name='conv2_pad')
            self.conv2 = layers.Conv2D(
                planes,
                kernel_size=3,
                strides=stride,
                padding='valid',
                dilation_rate=dilation,
                use_bias=False,
                data_format=data_format,
                name='conv2')

        conv3_in_planes = 2 * planes if factorize else planes
        self.conv3 = layers.Conv2D(
            planes * self.expansion,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name='conv3')
        self.bn3 = layers.BatchNormalization(axis=bn_axis, name='bn3')

        self.relu_out = layers.ReLU(name='relu_out')
        self.downsample = downsample

    def _inner_forward(self, x, training=False):
        """Forward wrapper for utilizing checkpoint."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)

        if not self.factorize:
            out = self.conv2_padding(out)
        out = self.conv2(out, training=training)

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


def make_res_layer_tf(block,
                      in_planes,
                      planes,
                      blocks,
                      stride=1,
                      dilation=1,
                      factorize=1,
                      with_cp=False,
                      data_format='channels_first',
                      name=None):
    """Build a residual layer for ResNetAudio."""
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = Sequential(
            [
                layers.Conv2D(
                    planes * block.expansion,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False,
                    data_format=data_format),
                layers.BatchNormalization(
                    axis=1 if data_format == 'channels_first' else -1)
            ],
            name='downsample')

    factorize_stages = factorize if not isinstance(
        factorize, int) else (factorize, ) * blocks
    
    layers_list = []
    layers_list.append(
        block(
            in_planes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            factorize=(factorize_stages[0] == 1),
            with_cp=with_cp,
            data_format=data_format,
            name='block1'))
    in_planes = planes * block.expansion
    for i in range(1, blocks):
        layers_list.append(
            block(
                in_planes,
                planes,
                stride=1,
                dilation=dilation,
                factorize=(factorize_stages[i] == 1),
                with_cp=with_cp,
                data_format=data_format,
                name=f'block{i + 1}'))

    return Sequential(layers_list, name=name)


class ResNetAudioTF(Model):
    """TensorFlow implementation of ResNetAudio backbone."""

    arch_settings = {
        50: (Bottleneck2dAudioTF, (3, 4, 6, 3)),
        101: (Bottleneck2dAudioTF, (3, 4, 23, 3)),
        152: (Bottleneck2dAudioTF, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=1,
                 num_stages=4,
                 base_channels=32,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=9,
                 conv1_stride=1,
                 frozen_stages=-1,
                 factorize=(1, 1, 0, 0),
                 with_cp=False,
                 data_format='channels_first',
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
        self.strides = strides
        self.dilations = dilations
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.frozen_stages = frozen_stages
        self.stage_factorization = factorize
        self.with_cp = with_cp
        self.data_format = data_format
        self.zero_init_residual = zero_init_residual
        self.bn_axis = 1 if self.data_format == 'channels_first' else -1

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.in_planes = self.base_channels

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = make_res_layer_tf(
                self.block,
                self.in_planes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                factorize=self.stage_factorization[i],
                with_cp=self.with_cp,
                data_format=self.data_format,
                name=f'layer{i + 1}')
            self.in_planes = planes * self.block.expansion
            self.res_layers.append(res_layer)

    def _make_stem_layer(self):
        """Construct the stem layers."""
        # To mimic 'same' padding in PyTorch for odd kernel sizes
        padding = (self.conv1_kernel - 1) // 2
        
        self.conv1 = Sequential([
            layers.ZeroPadding2D(
                padding=(padding, padding), data_format=self.data_format),
            layers.Conv2D(
                self.base_channels,
                kernel_size=self.conv1_kernel,
                strides=self.conv1_stride,
                use_bias=False,
                data_format=self.data_format,
                name='conv1')
        ], name='conv1')

        self.bn1 = layers.BatchNormalization(
            axis=self.bn_axis, name='stem_bn')
        self.relu = layers.ReLU(name='stem_relu')

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        for res_layer in self.res_layers:
            x = res_layer(x, training=training)
        return x

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for m in self.layers:
            if isinstance(m, layers.Conv2D):
                # Keras default is 'glorot_uniform', which is similar to kaiming
                pass
            elif isinstance(m, layers.BatchNormalization):
                m.gamma.assign(tf.ones_like(m.gamma))
                m.beta.assign(tf.zeros_like(m.beta))
        
        if self.zero_init_residual:
            for m in self.layers:
                if isinstance(m, Bottleneck2dAudioTF):
                    m.bn3.gamma.assign(tf.zeros_like(m.bn3.gamma))

    def _freeze_stages(self):
        """Freeze stages during training."""
        if self.frozen_stages >= 0:
            self.conv1.trainable = False
            self.bn1.trainable = False
        
        for i in range(self.frozen_stages):
            self.res_layers[i].trainable = False 