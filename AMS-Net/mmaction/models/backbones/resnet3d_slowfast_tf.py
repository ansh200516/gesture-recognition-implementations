import tensorflow as tf
from tensorflow.keras import layers

# A placeholder for the registry, to be replaced with a proper TF-based one if available.
class Registry:
    def __init__(self, name):
        self._module_dict = dict()

    def register_module(self, name=None):
        def _register(cls):
            key = name if name is not None else cls.__name__
            self._module_dict[key] = cls
            return cls
        return _register

BACKBONES = Registry('backbone')

# Helper to build a Keras-like ConvModule.
# This is a simplified version.
def ConvModule(filters, kernel_size, strides, padding, use_bias, norm_cfg=None, act_cfg=None, name=None):
    seq = tf.keras.Sequential(name=name)
    if any(p > 0 for p in padding):
        seq.add(layers.ZeroPadding3D(padding))
    
    seq.add(layers.Conv3D(filters, kernel_size, strides=strides, padding='valid', use_bias=use_bias))

    if norm_cfg is not None:
        if norm_cfg['type'] == 'BN3d':
            seq.add(layers.BatchNormalization())
        else:
            raise NotImplementedError(f"Normalization type {norm_cfg['type']} not supported.")
    
    if act_cfg is not None:
        # Assuming ReLU for any activation config for simplicity
        seq.add(layers.ReLU())

    return seq


class ResNet3dPathway(ResNet3d_tf):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d_tf``.
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Default: False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Default: 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Default: 5.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d_tf.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=5,
                 **kwargs):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        
        # This call will use the overridden make_res_layer
        super().__init__(*args, **kwargs)

        if self.lateral:
            # Recreate lateral connections as TF layers
            self.conv1_lateral = ConvModule(
                filters=self.base_channels * 2 // self.channel_ratio,
                kernel_size=(self.fusion_kernel, 1, 1),
                strides=(self.speed_ratio, 1, 1),
                padding=((self.fusion_kernel - 1) // 2, 0, 0),
                use_bias=False,
                norm_cfg=None,
                act_cfg=None,
                name='conv1_lateral')

            self.lateral_connections_layers = []
            inplanes = self.base_channels
            for i in range(self.num_stages):
                planes = self.base_channels * 2**i
                inplanes = planes * self.block.expansion

                if i != self.num_stages - 1:
                    lateral_inplanes = inplanes * 2 // self.channel_ratio
                    layer = ConvModule(
                        filters=lateral_inplanes,
                        kernel_size=(self.fusion_kernel, 1, 1),
                        strides=(self.speed_ratio, 1, 1),
                        padding=((self.fusion_kernel - 1) // 2, 0, 0),
                        use_bias=False,
                        norm_cfg=None,
                        act_cfg=None,
                        name=f'layer{(i + 1)}_lateral')
                    self.lateral_connections_layers.append(layer)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       style='pytorch',
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       non_local_cfg=dict(),
                       with_cp=False):
        """Build residual layer for Slowfast."""
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(
            non_local, int) else (non_local, ) * blocks
        
        if self.lateral:
            lateral_inplanes = inplanes * 2 // self.channel_ratio
        else:
            lateral_inplanes = 0
        
        actual_inplanes = inplanes + lateral_inplanes
        
        if (spatial_stride != 1
                or actual_inplanes != planes * block.expansion):
            downsample = ConvModule(
                filters=planes * block.expansion,
                kernel_size=1,
                strides=(temporal_stride, spatial_stride, spatial_stride),
                padding=(0, 0, 0),
                use_bias=False,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        else:
            downsample = None

        layer_list = []
        layer_list.append(
            block(
                actual_inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                with_cp=with_cp))
        
        inplanes_block = planes * block.expansion
        for i in range(1, blocks):
            layer_list.append(
                block(
                    inplanes_block,
                    planes,
                    1,
                    1,
                    dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    with_cp=with_cp))

        return tf.keras.Sequential(layer_list)


def build_pathway(cfg, *args, **kwargs):
    """Build pathway for SlowFast in TensorFlow."""
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    if pathway_type == 'resnet3d':
        pathway_cls = ResNet3dPathway
    else:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')
    
    # Assuming the base ResNet3d_tf and its variants are Keras models
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@BACKBONES.register_module()
class ResNet3dSlowFast(tf.keras.Model):
    """Slowfast backbone in TensorFlow.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_
    """

    def __init__(self,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 slow_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1),
                 **kwargs):
        super().__init__(**kwargs)
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio
            slow_pathway['channel_ratio'] = channel_ratio

        self.slow_path = build_pathway(slow_pathway)
        self.fast_path = build_pathway(fast_pathway)

    def call(self, x, training=False):
        """Defines the computation performed at every call."""
        # Assuming data_format is 'N, T, H, W, C'
        # Slow pathway: sparse sampling
        x_slow = x[:, ::self.resample_rate, :, :, :]
        
        # Fast pathway: dense sampling
        alpha = self.speed_ratio
        tau = self.resample_rate
        x_fast = x[:, ::(tau // alpha), :, :, :]

        x_slow = self.slow_path.conv1(x_slow, training=training)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = self.fast_path.conv1(x_fast, training=training)
        x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast, training=training)
            x_slow = tf.concat([x_slow, x_fast_lateral], axis=-1)

        for i in range(self.slow_path.num_stages):
            slow_res_layer = self.slow_path.res_layers[i]
            x_slow = slow_res_layer(x_slow, training=training)

            fast_res_layer = self.fast_path.res_layers[i]
            x_fast = fast_res_layer(x_fast, training=training)
            
            if (self.slow_path.lateral and i < self.slow_path.num_stages -1):
                lateral_conv = self.slow_path.lateral_connections_layers[i]
                x_fast_lateral = lateral_conv(x_fast, training=training)
                x_slow = tf.concat([x_slow, x_fast_lateral], axis=-1)

        return x_slow, x_fast

# The original file had a check for mmdet and registered the backbone there too.
# This might not be relevant for a pure TF version unless a similar ecosystem exists.
# try:
#     from mmdet.models import BACKBONES as MMDET_BACKBONES
#     if MMDET_BACKBONES:
#         MMDET_BACKBONES.register_module()(ResNet3dSlowFast)
# except (ImportError, ModuleNotFoundError):
#     pass

try:
    from .resnet3d_tf import ResNet3d as ResNet3d_tf
except ImportError:
    print("Warning: Could not import ResNet3d_tf. Using a dummy class.")
    class ResNet3d_tf(tf.keras.Model):
        def __init__(self, *args, **kwargs):
            super().__init__()
            for key, value in kwargs.items():
                setattr(self, key, value)
            
            # Dummy layers for ResNet3dPathway to work
            self.block = lambda *args, **kwargs: tf.keras.Sequential([layers.Dense(32)]) 
            self.num_stages = 4
            self.base_channels = 64
            self.norm_cfg = {'type': 'BN3d'}
            self.res_layers = []

            # Create dummy res_layers
            for i in range(self.num_stages):
                res_layer = self.make_res_layer(self.block, 64, 64, 2)
                self.res_layers.append(res_layer)
        
        def make_res_layer(self, block, inplanes, planes, blocks, **kwargs):
             return tf.keras.Sequential([block(inplanes, planes, **kwargs) for _ in range(blocks)]) 