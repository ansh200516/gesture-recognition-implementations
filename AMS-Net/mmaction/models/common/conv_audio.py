import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec


class ConvAudio(Conv):
    """ConvAudio layer for AudioResNet backbone.
    This layer performs a convolution on the input audio data.
    Args:
        rank (int): An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        filters (int): An integer, the dimensionality of the output space (i.e. the number of filters in the
                        convolution).
        kernel_size (int or tuple/list of 2 integers): Specifies the height and width of the 2D convolution
                                                        window. Can be a single integer to specify the same value for
                                                        all spatial dimensions.
        op (str): Operation to merge the output of freq and time feature map. Choices are 'sum' and 'concat'.
                    Default: 'concat'.
        strides (int or tuple/list of 2 integers): Specifies the strides of the convolution along the height and
                                                    width. Can be a single integer to specify the same value for all
                                                    spatial dimensions.
        padding (str): One of "valid" or "same" (case-insensitive).
        data_format (str): One of "channels_last" (default) or "channels_first". The ordering of the
                            dimensions in the inputs. "channels_last" corresponds to inputs with shape
                            (batch, height, width, channels) while "channels_first" corresponds to
                            inputs with shape (batch, channels, height, width).
        dilation_rate (int or tuple/list of 2 integers): Specifies the dilation rate to use for dilated convolution.
                                                        Can be a single integer to specify the same value for all
                                                        spatial dimensions.
        activation (callable): Activation function to use. If you don't specify anything, no activation is applied
                                (see keras.activations).
        use_bias (bool): Boolean, whether the layer uses a bias.
        kernel_initializer (callable): An initializer for the convolution kernel.
        bias_initializer (callable): An initializer for the bias vector. If None, the default initializer will be used.
        kernel_regularizer (callable): A regularizer instance for the convolution kernel.
        bias_regularizer (callable): A regularizer instance for the bias vector.
        activity_regularizer (callable): A regularizer instance for the output.
        kernel_constraint (callable): A constraint function for the kernel.
        bias_constraint (callable): A constraint function for the bias vector.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 op='concat',
                 rank=2,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ConvAudio,
              self).__init__(rank, filters, kernel_size, strides, padding,
                             data_format, dilation_rate, activation, use_bias,
                             kernel_initializer, bias_initializer,
                             kernel_regularizer, bias_regularizer,
                             activity_regularizer, kernel_constraint,
                             bias_constraint, **kwargs)
        self.op = op

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.kernel_1 = self.add_weight(
            name='kernel_1',
            shape=(self.kernel_size[0], 1) + (input_dim, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        self.kernel_2 = self.add_weight(
            name='kernel_2',
            shape=(1, self.kernel_size[1]) + (input_dim, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias_1 = self.add_weight(
                name='bias_1',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
            self.bias_2 = self.add_weight(
                name='bias_2',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias_1 = None
            self.bias_2 = None
        self.input_spec = InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        padding_1 = conv_utils.normalize_padding_2d(
            (self.kernel_size[0] // 2, 0))
        inputs_1 = tf.pad(inputs,
                          [[0, 0], [padding_1[0][0], padding_1[0][1]],
                           [padding_1[1][0], padding_1[1][1]], [0, 0]])
        outputs_1 = tf.nn.convolution(
            inputs_1,
            self.kernel_1,
            strides=list(self.strides),
            padding='VALID',
            data_format=conv_utils.convert_data_format(
                self.data_format, self.rank + 2),
            dilations=list(self.dilation_rate))

        padding_2 = conv_utils.normalize_padding_2d(
            (0, self.kernel_size[1] // 2))
        inputs_2 = tf.pad(inputs,
                          [[0, 0], [padding_2[0][0], padding_2[0][1]],
                           [padding_2[1][0], padding_2[1][1]], [0, 0]])
        outputs_2 = tf.nn.convolution(
            inputs_2,
            self.kernel_2,
            strides=list(self.strides),
            padding='VALID',
            data_format=conv_utils.convert_data_format(
                self.data_format, self.rank + 2),
            dilations=list(self.dilation_rate))

        if self.use_bias:
            outputs_1 = tf.nn.bias_add(
                outputs_1,
                self.bias_1,
                data_format=conv_utils.convert_data_format(self.data_format, 4))
            outputs_2 = tf.nn.bias_add(
                outputs_2,
                self.bias_2,
                data_format=conv_utils.convert_data_format(self.data_format, 4))

        if self.op == 'concat':
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
            outputs = tf.concat([outputs_1, outputs_2], channel_axis)
        else:
            outputs = outputs_1 + outputs_2

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {'op': self.op}
        base_config = super(ConvAudio, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
