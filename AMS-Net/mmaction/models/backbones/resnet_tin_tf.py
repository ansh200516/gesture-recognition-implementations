import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from ..registry import BACKBONES
from .resnet_tsm_tf import ResNetTSM_TF


def tin_shift(data, offset):
    """
    Performs a temporal shift on the data tensor based on the given offset.
    This is a TensorFlow implementation of a functionality similar to mmcv.ops.tin_shift.
    It uses tf.gather_nd to select frames from different temporal locations.
    """
    # data: (N, T, C, HW), offset: (N, T_new)
    n = tf.shape(data)[0]
    t = tf.shape(data)[1]
    c = data.get_shape().as_list()[2]
    hw = data.get_shape().as_list()[3]
    t_new = tf.shape(offset)[1]

    offset = tf.clip_by_value(offset, 0, t - 1)
    
    batch_indices = tf.tile(tf.reshape(tf.range(n), [n, 1, 1]), [1, t_new, 1])
    temporal_indices = tf.reshape(offset, [n, t_new, 1])
    
    indices = tf.concat([batch_indices, temporal_indices], axis=2)
    
    shifted_data = tf.gather_nd(data, indices)
    return shifted_data


def linear_sampler(data, offset):
    """
    Differentiable Temporal-wise Frame Sampling in TensorFlow.
    It's a linear interpolation process.
    """
    # data: [N, T, C, H, W], offset: [N, T_new]
    n, t, c, h, w = data.get_shape().as_list()
    t_new = tf.shape(offset)[1]
    
    data_reshaped = tf.reshape(data, [n, t, c, h * w])

    offset0 = tf.floor(offset)
    offset1 = offset0 + 1

    offset0 = tf.cast(offset0, tf.int32)
    offset1 = tf.cast(offset1, tf.int32)
    
    data0 = tin_shift(data_reshaped, offset0)
    data1 = tin_shift(data_reshaped, offset1)

    weight0 = 1.0 - (offset - tf.cast(offset0, tf.float32))
    weight1 = 1.0 - weight0
    
    weight0 = weight0[:, :, tf.newaxis]
    weight1 = weight1[:, :, tf.newaxis]

    output = weight0 * data0 + weight1 * data1
    output = tf.reshape(output, [n, t_new, c, h, w])

    return output


class CombineNet(Model):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def call(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


class WeightNet(Model):
    def __init__(self, in_channels, groups):
        super().__init__()
        self.groups = groups
        self.conv = layers.Conv1D(filters=groups, kernel_size=3, padding='same', bias_initializer='zeros')
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        # x: [N, C, T] -> [N, T, C] for Conv1D
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.conv(x)
        # x: [N, T, groups]
        x = 2 * self.sigmoid(x)
        return x


class OffsetNet(Model):
    def __init__(self, in_channels, groups, num_segments):
        super().__init__()
        self.conv = layers.Conv1D(filters=1, kernel_size=3, padding='same')
        self.fc1 = layers.Dense(num_segments, activation='relu')
        self.fc2 = layers.Dense(groups, bias_initializer=tf.keras.initializers.Constant(0.5108))
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        # x: [N, C, T] -> [N, T, C] for Conv1D
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.conv(x)
        x = tf.squeeze(x, axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x[:, tf.newaxis, :]
        x = 4 * (self.sigmoid(x) - 0.5)
        return x


class TemporalInterlace(Model):
    def __init__(self, in_channels, num_segments=3, shift_div=1):
        super().__init__()
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.in_channels = in_channels
        self.deform_groups = 2

        self.offset_net = OffsetNet(in_channels // shift_div, self.deform_groups, num_segments)
        self.weight_net = WeightNet(in_channels // shift_div, self.deform_groups)

    def call(self, x):
        # x: [N, C, H, W] where N = num_batches * num_segments
        n, c, h, w = x.get_shape().as_list()
        num_batches = n // self.num_segments
        num_folds = c // self.shift_div

        x_descriptor = tf.reshape(x[:, :num_folds, :, :], [num_batches, self.num_segments, num_folds, h, w])

        x_pooled = tf.reduce_mean(x_descriptor, axis=3)
        x_pooled = tf.reduce_mean(x_pooled, axis=3)
        x_pooled = tf.transpose(x_pooled, perm=[0, 2, 1])

        x_offset = self.offset_net(x_pooled)
        x_offset = tf.reshape(x_offset, [num_batches, -1])

        x_weight = self.weight_net(x_pooled)
        
        x_offset = tf.tile(x_offset, [1, self.num_segments // self.deform_groups])

        x_shift = linear_sampler(x_descriptor, x_offset)

        x_weight = tf.reshape(x_weight, [num_batches, self.num_segments, 1])
        x_weight = x_weight[:, :, :, tf.newaxis, tf.newaxis]
        x_shift = x_shift * x_weight
        x_shift = tf.reshape(x_shift, [n, num_folds, h, w])

        x_out = tf.concat([x_shift, x[:, num_folds:, :, :]], axis=1)
        return x_out


@BACKBONES.register_module()
class ResNetTIN_TF(ResNetTSM_TF):
    def __init__(self,
                 depth,
                 num_segments=8,
                 is_tin=True,
                 shift_div=4,
                 **kwargs):
        super().__init__(depth=depth, num_segments=num_segments, **kwargs)
        self.is_tin = is_tin
        self.shift_div = shift_div

        if self.is_tin:
            self.make_temporal_interlace()

    def make_temporal_interlace(self):
        n_round = 1
        
        def make_block_interlace(stage, num_segments, shift_div):
            blocks = list(stage.layers)
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    in_channels = b.conv1.conv.in_channels
                    tds = TemporalInterlace(in_channels, num_segments=num_segments, shift_div=shift_div)
                    blocks[i].conv1.conv = CombineNet(tds, blocks[i].conv1.conv)
            return Sequential(blocks)

        self.layer1 = make_block_interlace(self.layer1, self.num_segments, self.shift_div)
        self.layer2 = make_block_interlace(self.layer2, self.num_segments, self.shift_div)
        self.layer3 = make_block_interlace(self.layer3, self.num_segments, self.shift_div)
        self.layer4 = make_block_interlace(self.layer4, self.num_segments, self.shift_div) 