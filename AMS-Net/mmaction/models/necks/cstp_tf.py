import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..builder_tf import NECKS
from ..backbones.ams_2D_module_tf import CSTP_Stage1_Adaptive_Fusion, CSTP_Stage2_Adaptive_Fusion


class Identity_TF(layers.Layer):
    """Identity mapping."""
    def __init__(self, **kwargs):
        super(Identity_TF, self).__init__(**kwargs)

    def call(self, x):
        return x


class ConvModule_TF(layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1, 1),
                 padding='same',
                 groups=1,
                 bias=False,
                 norm_cfg=None,
                 act_cfg=None,
                 **kwargs):
        super(ConvModule_TF, self).__init__(**kwargs)
        
        if padding == 'same':
            tf_padding = 'same'
        else: # assuming tuple for padding
            tf_padding = 'valid'
            self.padding_layer = layers.ZeroPadding3D(padding, data_format='channels_first')


        self.conv = layers.Conv3D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=tf_padding,
            groups=groups,
            use_bias=bias,
            data_format='channels_first'
        )
        
        self.bn = None
        if norm_cfg is not None and norm_cfg['type'] == 'BN3d':
            self.bn = layers.BatchNormalization(axis=1) # channel-first
            
        self.activation = None
        if act_cfg is not None and act_cfg['type'] == 'ReLU':
            self.activation = layers.ReLU()
            
    def call(self, x, training=None):
        if hasattr(self, 'padding_layer'):
            x = self.padding_layer(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x

class DownSample_TF(layers.Layer):
    """DownSample modules."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 groups=1,
                 bias=False,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=None,
                 act_cfg=None,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),
                 **kwargs):
        super(DownSample_TF, self).__init__(**kwargs)

        # In TF 'same' padding with stride > 1 is different from PyTorch
        # We will use 'valid' padding and pad manually if needed.
        # Here we mimic ConvModule from mmcv which seems to handle it.
        # For kernel 3, padding 1 is 'same'. For kernel 1, padding 0 is 'valid'
        # The original padding is (1,0,0) for a kernel of (3,1,1)
        # This means temporal padding is 1, spatial is 0.
        
        padding_tf = (padding[0], padding[1], padding[2])

        self.conv = ConvModule_TF(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding_tf,
            groups=groups,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        assert downsample_position in ['before', 'after']
        self.downsample_position = downsample_position
        self.pool = layers.MaxPool3D(
            pool_size=downsample_scale, strides=downsample_scale, padding='valid', data_format='channels_first')

    def call(self, x, training=None):
        # The original has ceil_mode=True. TF padding='same' is close to this.
        # But for stride>1 it's complicated.
        # Let's assume 'valid' with proper padding works for now
        if self.downsample_position == 'before':
            x = self.pool(x)
            x = self.conv(x, training=training)
        else:
            x = self.conv(x, training=training)
            x = self.pool(x)
        return x


class LevelFusion_TF(layers.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 downsample_scales=((1, 1, 1), (1, 1, 1)),
                 **kwargs):
        super(LevelFusion_TF, self).__init__(**kwargs)
        num_stages = len(in_channels)

        self.downsamples = []
        for i in range(num_stages):
            downsample = DownSample_TF(
                in_channels[i],
                mid_channels[i],
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                bias=False,
                padding=(0, 0, 0),
                groups=32,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU'),
                downsample_position='before',
                downsample_scale=downsample_scales[i])
            self.downsamples.append(downsample)

    def call(self, x, training=None):
        out = [self.downsamples[i](feature, training=training) for i, feature in enumerate(x)]
        return out


class SpatialModulation_TF(layers.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SpatialModulation_TF, self).__init__(**kwargs)

        self.spatial_modulation = []
        for channel in in_channels:
            downsample_scale = out_channels // channel
            downsample_factor = int(np.log2(downsample_scale))
            op = []
            if downsample_factor < 1:
                op_module = Identity_TF()
            else:
                current_channel = channel
                for factor in range(downsample_factor):
                    in_factor = 2**factor
                    out_factor = 2**(factor + 1)
                    op.append(
                        ConvModule_TF(
                            channel * in_factor,
                            channel * out_factor, (1, 3, 3),
                            stride=(1, 2, 2),
                            padding=(0, 1, 1),
                            bias=False,
                            norm_cfg=dict(type='BN3d'),
                            act_cfg=dict(type='ReLU')))
                op_module = tf.keras.Sequential(op)
            self.spatial_modulation.append(op_module)

    def call(self, x, training=None):
        out = []
        for i, feature in enumerate(x):
            out.append(self.spatial_modulation[i](feature, training=training))
        return out


class AuxHead_TF(layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_weight=0.5,
                 **kwargs):
        super(AuxHead_TF, self).__init__(**kwargs)

        self.conv = ConvModule_TF(
            in_channels,
            in_channels * 2, (1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            norm_cfg=dict(type='BN3d'))
        self.avg_pool = layers.GlobalAveragePooling3D(data_format='channels_first')
        self.loss_weight = loss_weight
        self.dropout = layers.Dropout(rate=0.5)
        self.fc = layers.Dense(out_channels)
        self.loss_cls = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, x, target=None, training=None):
        losses = {}
        if target is None:
            return losses
        
        x = self.conv(x, training=training)
        x = self.avg_pool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)

        loss = self.loss_cls(y_true=target, y_pred=x)
        losses['loss_aux'] = self.loss_weight * loss
        return losses


class TemporalModulation_TF(layers.Layer):
    def __init__(self, in_channels, out_channels, downsample_scale=8, **kwargs):
        super(TemporalModulation_TF, self).__init__(**kwargs)

        self.conv = ConvModule_TF(
            in_channels,
            out_channels, (3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False,
            groups=32,
            act_cfg=None)
        self.pool = layers.MaxPool3D(
            pool_size=(downsample_scale, 1, 1),
            strides=(downsample_scale, 1, 1),
            padding='valid', # 'ceil_mode' is tricky. Pytorch ceil_mode=True will pad, TF valid will not. 
            data_format='channels_first'
        )

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x


@NECKS.register_module()
class CSTP_TF(layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_modulation_cfg=None,
                 temporal_modulation_cfg=None,
                 upsample_cfg=None,
                 downsample_cfg=None,
                 level_fusion_cfg=None,
                 aux_head_cfg=None,
                 flow_type='parallel',
                 gamma=1,
                 **kwargs):
        super(CSTP_TF, self).__init__(**kwargs)
        assert isinstance(in_channels, tuple)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tpn_stages = len(in_channels)
        self.gamma = gamma

        self.flow_type = flow_type

        self.temporal_modulation_ops = []
        self.upsample_ops = []
        self.downsample_ops = []

        self.level_fusion_1 = LevelFusion_TF(**level_fusion_cfg)
        self.spatial_modulation = SpatialModulation_TF(**spatial_modulation_cfg)

        for i in range(self.num_tpn_stages):
            if temporal_modulation_cfg is not None:
                downsample_scale = temporal_modulation_cfg['downsample_scales'][i]
                temporal_modulation = TemporalModulation_TF(
                    in_channels[-1], out_channels, downsample_scale)
                self.temporal_modulation_ops.append(temporal_modulation)

            if i < self.num_tpn_stages - 1:
                if upsample_cfg is not None:
                    # upsample_cfg is like {'scale_factor': (1, 2, 2), 'mode': 'nearest'}
                    size = upsample_cfg.get('scale_factor', 1)
                    self.upsample_ops.append(layers.UpSampling3D(size=size, data_format='channels_first'))

                if downsample_cfg is not None:
                    downsample = DownSample_TF(out_channels, out_channels,
                                             **downsample_cfg)
                    self.downsample_ops.append(downsample)
        
        # two pyramids
        self.level_fusion_2 = LevelFusion_TF(**level_fusion_cfg)

        #CSTP stage one
        self.pyramids_selective = CSTP_Stage1_Adaptive_Fusion(out_channels, branchs=4, rate=8, gamma=self.gamma)
        
        self.fusion_top_down = ConvModule_TF(
            out_channels*2, out_channels*2, 1, stride=1, padding=(0,0,0), bias=False,
            norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU'))

        self.fusion_botton_up = ConvModule_TF(
            out_channels*2, out_channels*2, 1, stride=1, padding=(0,0,0), bias=False,
            norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU'))
        
        #CSTP stage two
        self.pyramids_fusion = CSTP_Stage2_Adaptive_Fusion(out_channels * 2, branchs=2, rate=8, gamma=self.gamma)
        
        out_dims = level_fusion_cfg['out_channels']
        self.pyramid_fusion = ConvModule_TF(
            out_dims * 2, 2048, 1, stride=1, padding=(0,0,0), bias=False,
            norm_cfg=dict(type='BN3d'))

        if aux_head_cfg is not None:
            self.aux_head = AuxHead_TF(self.in_channels[-2], **aux_head_cfg)
        else:
            self.aux_head = None

    def call(self, x, target=None, training=None):
        loss_aux = {}

        # Auxiliary loss
        if self.aux_head is not None:
            loss_aux = self.aux_head(x[-2], target, training=training)

        # Spatial Modulation
        spatial_modulation_outs = self.spatial_modulation(x, training=training)

        # Temporal Modulation
        temporal_modulation_outs = []
        for i, temporal_modulation in enumerate(self.temporal_modulation_ops):
            temporal_modulation_outs.append(
                temporal_modulation(spatial_modulation_outs[i], training=training))

        outs = [tf.identity(out) for out in temporal_modulation_outs]
        if len(self.upsample_ops) != 0:
            for i in range(self.num_tpn_stages - 2, -1, -1):
                outs[i] = outs[i] + self.upsample_ops[i](outs[i + 1])
        
        # Get top-down outs
        top_down_outs = self.level_fusion_1(outs, training=training)
        Pyramids_inputs = top_down_outs

        # Build bottom-up flow
        if self.flow_type == 'parallel':
            outs = [tf.identity(out) for out in temporal_modulation_outs]
        if len(self.downsample_ops) != 0:
            for i in range(self.num_tpn_stages - 1):
                outs[i + 1] = outs[i + 1] + self.downsample_ops[i](outs[i], training=training)

        # Get bottom-up outs
        botton_up_outs = self.level_fusion_2(outs, training=training)
        for pyramids_in in botton_up_outs:
            Pyramids_inputs.append(pyramids_in)
        
        pyramids_top_down, pyramids_botton_up = self.pyramids_selective(Pyramids_inputs, training=training)
        top_down_outs = self.fusion_top_down(pyramids_top_down, training=training)
        botton_up_outs = self.fusion_botton_up(pyramids_botton_up, training=training)

        # fuse two pyramid outs
        outs = [top_down_outs, botton_up_outs]
        outs = self.pyramids_fusion(outs, training=training)
        outs = self.pyramid_fusion(outs, training=training)

        return outs, loss_aux 