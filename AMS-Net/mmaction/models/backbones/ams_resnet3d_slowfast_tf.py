import tensorflow as tf
from tensorflow.keras import layers

from .ams_resnet3d_tf import AMSResNet3dTF
from .ams_3D_module_tf import CPTM_Bottleneck as AFG_module

# A tf.keras.layers.Layer that mimics mmcv.cnn.ConvModule's basic functionality.
class ConvModule(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_val = padding
        self.use_bias = use_bias
        self.norm_layer = norm_layer
        self.activation = activation

        self.padding_layer = None
        if isinstance(self.padding_val, (tuple, list)):
            pad_d, pad_h, pad_w = self.padding_val
            self.padding_layer = layers.ZeroPadding3D(
                padding=((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)))
            conv_padding = 'valid'
        else:
            conv_padding = self.padding_val
        
        self.conv = layers.Conv3D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=conv_padding,
            use_bias=self.use_bias
        )

    def call(self, inputs, training=None):
        x = inputs
        if self.padding_layer:
            x = self.padding_layer(x)
        x = self.conv(x)
        if self.norm_layer:
            x = self.norm_layer(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

class AMSResNet3dPathway(AMSResNet3dTF):
    """A pathway of Slowfast based on ResNet3d in TensorFlow.

    Args:
        *args (arguments): Arguments same as :class:``AMSResNet3dTF``.
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
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
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
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels

        if self.lateral:
            self.conv1_lateral = ConvModule(
                filters=self.inplanes * 2 // self.channel_ratio,
                kernel_size=(fusion_kernel, 1, 1),
                strides=(self.speed_ratio, 1, 1),
                padding=((fusion_kernel - 1) // 2, 0, 0),
                use_bias=False,
                norm_layer=None,
                activation=None)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                lateral_conv = ConvModule(
                    filters=self.inplanes * 2 // self.channel_ratio,
                    kernel_size=(fusion_kernel, 1, 1),
                    strides=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    use_bias=False,
                    norm_layer=None,
                    activation=None)
                setattr(self, lateral_name, lateral_conv)
                self.lateral_connections.append(lateral_name)

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
                       conv_cfg=None,
                       norm_cfg=None,
                       act_cfg=None,
                       with_cp=False):
        """Build residual layer for Slowfast."""
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(
            non_local, int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks

        if self.lateral:
            lateral_inplanes = inplanes * 2 // self.channel_ratio
        else:
            lateral_inplanes = 0

        if (spatial_stride != 1
                or (inplanes + lateral_inplanes) != planes * block.expansion):
            downsample = ConvModule(
                filters=planes * block.expansion,
                kernel_size=1,
                strides=(temporal_stride, spatial_stride, spatial_stride),
                use_bias=False,
                norm_layer=layers.BatchNormalization() if norm_cfg else None,
                activation=None)
        else:
            downsample = None

        res_layers = []
        res_layers.append(
            block(
                inplanes + lateral_inplanes,
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
                with_cp=with_cp,
                AMG_dim_ratio=0.5,
            ))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            res_layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    1,
                    dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    with_cp=with_cp,
                    last_block=(i == blocks - 1),
                    insert_stage_flag=(
                        (planes == 256 and i % 2 == 1) or
                        (planes == 512 and i == blocks - 1))))
        
        return tf.keras.Sequential(res_layers)

    def call(self, x, lateral_features=None, training=None):
        # This is a simplified call function. The original SlowFast model would
        # handle the interaction between fast and slow pathways.
        # Here we assume `x` is the input to this pathway, and `lateral_features`
        # are provided from the other pathway if `self.lateral` is True.

        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if self.lateral and lateral_features:
            lateral_conv1 = lateral_features.pop(0)
            lateral_res = self.conv1_lateral(lateral_conv1, training=training)
            x = tf.concat([x, lateral_res], axis=-1)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x, training=training)
            if self.lateral and lateral_features and i < len(self.lateral_connections):
                 lateral_conv = getattr(self, self.lateral_connections[i])
                 lateral_res = lateral_conv(lateral_features.pop(0), training=training)
                 x = tf.concat([x, lateral_res], axis=-1)

        return x

    def inflate_weights(self, logger):
        """Placeholder for weight inflation from a 2D pre-trained model."""
        print("Weight inflation from 2D checkpoint is not implemented in this TF version.")
        pass

    def init_weights(self, pretrained=None):
        """Placeholder for weight initialization."""
        if pretrained:
            self.pretrained = pretrained
            # Weight loading logic would go here.
            print(f"Pretrained model loading from {pretrained} is not implemented.")
        else:
            # Standard Keras initializers are used by default.
            # Custom initialization can be added here.
            pass

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.trainable = False
            self.bn1.trainable = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.trainable = False

            if i <= len(self.lateral_connections) and self.lateral:
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.trainable = False


def build_pathway(cfg, *args, **kwargs):
    """Build pathway for SlowFast in TensorFlow.

    Args:
        cfg (None or dict): cfg should contain:
            - type (str): identify conv layer type.

    Returns:
        tf.keras.Model: Created pathway.
    """
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    
    # In this TF version, we only have one pathway type.
    # A more robust implementation might have a registry like in PyTorch.
    if pathway_type not in ['resnet3d']:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')

    pathway_cls = AMSResNet3dPathway
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway 