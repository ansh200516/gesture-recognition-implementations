import tensorflow as tf
from tensorflow.keras import layers

# TODO: Need to import ResNet from a TF implementation
# from .resnet_tf import ResNet
from tensorflow.keras.applications import ResNet50 as ResNet # Placeholder

# TODO: Need to define BACKBONES registry for TF
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self):
        def _register(cls):
            self._module_dict[cls.__name__] = cls
            return cls
        return _register
    
    def get(self, key):
        return self._module_dict.get(key)

BACKBONES = Registry('backbone')


class TemporalShift(layers.Layer):
    """Temporal shift module.
    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_
    Args:
        num_segments (int): Number of frame segments. Default: 8.
        shift_div (int): Number of divisions for shift. Default: 8.
    """
    def __init__(self, num_segments=8, shift_div=8, **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.shift_div = shift_div

    def call(self, x):
        return self.shift(x, self.num_segments, self.shift_div)

    @staticmethod
    def shift(x, num_segments, shift_div=8):
        """Perform temporal shift operation on the feature.
        Args:
            x (tf.Tensor): The input feature to be shifted.
                (N, H, W, C)
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 8.
        Returns:
            tf.Tensor: The shifted feature.
        """
        shape = tf.shape(x)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        
        # Reshape to (B, T, H, W, C) from (B*T, H, W, C)
        x_reshaped = tf.reshape(x, [-1, num_segments, h, w, c])
        
        fold = c // shift_div
        
        left_split = x_reshaped[:, :, :, :, :fold]
        mid_split = x_reshaped[:, :, :, :, fold:2 * fold]
        right_split = x_reshaped[:, :, :, :, 2 * fold:]
        
        # Shift left on num_segments channel in `left_split`
        left_shifted = tf.concat([left_split[:, 1:, ...], tf.zeros_like(left_split[:, :1, ...])], axis=1)
        
        # Shift right on num_segments channel in `mid_split`
        mid_shifted = tf.concat([tf.zeros_like(mid_split[:, :1, ...]), mid_split[:, :-1, ...]], axis=1)
        
        # Concatenate
        out = tf.concat([left_shifted, mid_shifted, right_split], axis=4)
        
        # Restore the original dimension (N, H, W, C)
        return tf.reshape(out, (n, h, w, c))


class NonLocal3D(layers.Layer):
    def __init__(self, in_channels, inter_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        self.g = layers.Conv3D(self.inter_channels, kernel_size=1, strides=1, padding='same')
        self.theta = layers.Conv3D(self.inter_channels, kernel_size=1, strides=1, padding='same')
        self.phi = layers.Conv3D(self.inter_channels, kernel_size=1, strides=1, padding='same')
        
        self.conv_out = layers.Conv3D(self.in_channels, kernel_size=1, strides=1, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, x, training=None):
        # x shape: (N, T, H, W, C)
        batch_size = tf.shape(x)[0]

        g_x = self.g(x)
        g_x_shape = tf.shape(g_x)
        g_x = tf.reshape(g_x, (batch_size, g_x_shape[1] * g_x_shape[2] * g_x_shape[3], self.inter_channels))

        theta_x = self.theta(x)
        theta_x_shape = tf.shape(theta_x)
        theta_x = tf.reshape(theta_x, (batch_size, theta_x_shape[1] * theta_x_shape[2] * theta_x_shape[3], self.inter_channels))

        phi_x = self.phi(x)
        phi_x_shape = tf.shape(phi_x)
        phi_x = tf.reshape(phi_x, (batch_size, phi_x_shape[1] * phi_x_shape[2] * phi_x_shape[3], self.inter_channels))
        phi_x = tf.transpose(phi_x, (0, 2, 1))

        f = tf.matmul(theta_x, phi_x)
        f = tf.nn.softmax(f, axis=-1)

        y = tf.matmul(f, g_x)
        
        input_shape = tf.shape(x)
        y = tf.reshape(y, (batch_size, input_shape[1], input_shape[2], input_shape[3], self.inter_channels))

        y = self.conv_out(y)
        y = self.bn(y, training=training)

        return x + y


class NL3DWrapper(layers.Layer):
    def __init__(self, block, num_segments, non_local_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.block = block
        self.num_segments = num_segments
        self.non_local_cfg = non_local_cfg if non_local_cfg else {}
        # NonLocal3D is initialized in build

    def build(self, input_shape):
        c_out = self.block.compute_output_shape(input_shape)[-1]
        self.non_local_block = NonLocal3D(c_out, **self.non_local_cfg)
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.block(x, training=training)
        
        shape = tf.shape(x)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        
        x_reshaped = tf.reshape(x, [-1, self.num_segments, h, w, c])
        
        x_nl = self.non_local_block(x_reshaped, training=training)
        
        return tf.reshape(x_nl, (n, h, w, c))

class TemporalPool(layers.Layer):
    def __init__(self, net, num_segments, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.num_segments = num_segments
        self.max_pool3d = layers.MaxPool3D(
            pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same', data_format='channels_first')
    
    def call(self, x, training=None):
        shape = tf.shape(x)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        
        # (B, T, H, W, C) -> (B, C, T, H, W)
        x_reshaped = tf.reshape(x, [-1, self.num_segments, h, w, c])
        x_transposed = tf.transpose(x_reshaped, (0, 4, 1, 2, 3))

        x_pooled = self.max_pool3d(x_transposed)
        
        # (B, C, T/2, H, W) -> (B, T/2, H, W, C) -> (B*T/2, H, W, C)
        x_out = tf.transpose(x_pooled, (0, 2, 3, 4, 1))
        
        out_shape = tf.shape(x_out)
        x_out = tf.reshape(x_out, (-1, out_shape[2], out_shape[3], out_shape[4]))

        return self.net(x_out, training=training)

@BACKBONES.register_module()
class ResNetTSM(tf.keras.Model):
    def __init__(self,
                 depth,
                 num_segments=8,
                 is_shift=True,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=None,
                 shift_div=8,
                 shift_place='blockres',
                 temporal_pool=False,
                 **kwargs):
        super().__init__(**kwargs)
        if depth not in [50, 101, 152]:
            raise NotImplementedError("Only ResNet-50, 101, 152 are currently supported for TSM.")
        
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.non_local_cfg = non_local_cfg if non_local_cfg else {}
        
        # Using tf.keras.applications.ResNet as a base
        # This is a placeholder for a custom ResNet implementation that matches mmaction's
        self.base_model = ResNet(depth=depth, include_top=False, weights=None, **kwargs)
        
        self.num_stages = 4 # For ResNet
        self.non_local = non_local
        
        if self.is_shift:
            self.make_temporal_shift()
        if any(sum(nl) for nl in self.non_local):
             self.make_non_local()
        if self.temporal_pool:
            self.make_temporal_pool()

    def make_temporal_shift(self):
        if self.temporal_pool:
            num_segment_list = [self.num_segments, self.num_segments // 2, self.num_segments // 2, self.num_segments // 2]
        else:
            num_segment_list = [self.num_segments] * 4

        if self.shift_place == 'block':
            raise NotImplementedError("shift_place='block' is not implemented in TF version yet.")
        
        elif 'blockres' in self.shift_place:
            # Modify ResNet blocks to include TemporalShift
            # This requires access to the internal layers of the ResNet model.
            # The logic below is a conceptual sketch and depends on the ResNet implementation details.
            
            # This is highly dependent on the structure of the ResNet implementation.
            # For tf.keras.applications.ResNet, modifying internal layers is complex.
            # A custom ResNet builder would be more suitable here.
            print("Warning: Temporal shift is conceptually applied. A custom ResNet allowing layer wrapping is needed for a real implementation.")
            pass # Placeholder for modification logic.

    def make_non_local(self):
        # Placeholder for non-local block insertion.
        # This also requires a flexible ResNet implementation.
        print("Warning: Non-local blocks are conceptually applied. A custom ResNet allowing layer wrapping is needed.")
        pass

    def make_temporal_pool(self):
        # Placeholder for temporal pooling.
        print("Warning: Temporal pooling is conceptually applied. A custom ResNet allowing layer wrapping is needed.")
        pass

    def call(self, x, training=None):
        # The forward pass will depend on how the modifications (shift, non-local, pool) are implemented.
        # A simple pass through the base_model is a placeholder.
        return self.base_model(x, training=training)

    def init_weights(self):
        # Weight initialization would be handled by Keras layers.
        # Loading pre-trained weights would be done here.
        pass 