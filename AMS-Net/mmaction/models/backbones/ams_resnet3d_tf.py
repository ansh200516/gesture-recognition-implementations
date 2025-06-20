import tensorflow as tf
from tensorflow.keras import layers

from .ams_3D_module_tf import CPTM_bottleneck


class ConvModule(layers.Layer):
    def __init__(self, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 padding, 
                 dilation, 
                 use_bias, 
                 data_format='channels_first',
                 norm='BN',
                 activation='relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.use_padding = any(p > 0 for p in padding)
        if self.use_padding:
            self.padding_layer = layers.ZeroPadding3D(padding=padding, data_format=data_format)

        self.conv = layers.Conv3D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding='valid' if self.use_padding else 'same',
            dilation_rate=dilation,
            use_bias=use_bias,
            data_format=data_format
        )
        
        if norm == 'BN':
            self.bn = layers.BatchNormalization(axis=1 if data_format == 'channels_first' else -1)
        else:
            self.bn = None

        if activation == 'relu':
            self.activation = layers.ReLU()
        else:
            self.activation = None

    def call(self, x, training=None):
        if self.use_padding:
            x = self.padding_layer(x)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x


class BasicBlock3d(layers.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        
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

        self.conv1 = ConvModule(
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            use_bias=False)

        self.conv2 = ConvModule(
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, 1, 1),
            use_bias=False,
            activation=None)
            
        self.downsample = downsample
        self.relu = layers.ReLU()

    def call(self, x, training=None):
        identity = x

        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3d(layers.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 inflate_style='3x1x1',
                 last_block=False,
                 insert_stage_flag=False,
                 AMG_dim_ratio=0.25,
                 **kwargs):
        super().__init__(**kwargs)
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.last_block = last_block
        self.insert_stage_flag = insert_stage_flag
        self.AMG_dim_ratio = AMG_dim_ratio
        
        self.conv1_stride_s = 1
        self.conv2_stride_s = spatial_stride
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1
        
        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else: # 3x3x3
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            use_bias=False)
            
        self.conv2 = ConvModule(
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            use_bias=False)
            
        self.conv3 = ConvModule(
            planes * self.expansion,
            kernel_size=(1, 1, 1),
            stride=(1,1,1),
            padding=(0,0,0),
            use_bias=False,
            activation=None)
            
        self.downsample = downsample
        self.relu = layers.ReLU()
        
        if self.insert_stage_flag:
            self.cptm = CPTM_bottleneck(planes * self.expansion, percent=self.AMG_dim_ratio)

    def call(self, x, training=None):
        identity = x
        
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        out = self.conv3(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)
        
        if self.insert_stage_flag:
            out = self.cptm(out, training=training)

        return out


class AMSResNet3d(tf.keras.Model):
    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(5, 7, 7),
                 conv1_stride_s=2,
                 conv1_stride_t=2,
                 pool1_stride_s=2,
                 pool1_stride_t=2,
                 with_pool2=True,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 frozen_stages=-1,
                 **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_stages = num_stages
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool2 = with_pool2
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.frozen_stages = frozen_stages
        
        self.block, self.stage_blocks = self.arch_settings[depth]
        self.inplanes = self.base_channels
        
        self._make_stem_layer()
        
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = self.spatial_strides[i]
            temporal_stride = self.temporal_strides[i]
            dilation = self.dilations[i]
            planes = self.base_channels * 2**i
            
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                inflate=self.inflate[i],
                inflate_style=self.inflate_style,
                name=f'layer{i+1}'
            )
            setattr(self, f'layer{i+1}', res_layer)
            self.res_layers.append(res_layer)
            self.inplanes = planes * self.block.expansion


    def make_res_layer(self, block, inplanes, planes, blocks, spatial_stride=1, temporal_stride=1, dilation=1, inflate=1, inflate_style='3x1x1', name=None):
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                padding=(0,0,0),
                use_bias=False,
                activation=None
            )
            
        layers = []
        layers.append(
            block(inplanes, 
                  planes, 
                  spatial_stride, 
                  temporal_stride, 
                  dilation,
                  downsample, 
                  inflate=inflate, 
                  inflate_style=inflate_style)
        )
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes,
                      planes,
                      1, 1,
                      dilation,
                      inflate=inflate,
                      inflate_style=inflate_style,
                      # Parameters for AMS module from original code
                      last_block=(i == blocks - 1),
                      insert_stage_flag = ((planes==256 and i%2==1) or (planes==512 and i==blocks-1))
                )
            )
        return tf.keras.Sequential(layers, name=name)


    def _make_stem_layer(self):
        self.conv1 = ConvModule(
            self.base_channels,
            self.conv1_kernel,
            (self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=(self.conv1_kernel[0] // 2, self.conv1_kernel[1] // 2, self.conv1_kernel[2] // 2),
            use_bias=False
        )
        self.pool1 = layers.MaxPool3D(
            pool_size=(1, self.pool1_stride_s, self.pool1_stride_s),
            strides=(1, self.pool1_stride_s, self.pool1_stride_s),
            padding='same',
            data_format='channels_first'
        )
        if self.with_pool2:
            self.pool2 = layers.MaxPool3D(
                pool_size=(self.pool1_stride_t, 1, 1),
                strides=(self.pool1_stride_t, 1, 1),
                padding='same',
                data_format='channels_first'
            )
            
    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.pool1(x)
        if self.with_pool2:
            x = self.pool2(x)
            
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x, training=training)
            if i in self.out_indices:
                outs.append(x)
        
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def init_weights(self):
        # Weight initialization logic would go here.
        # Keras layers have default initializers (e.g., Glorot uniform)
        # which are often sufficient.
        # For porting weights, one would load them here.
        pass

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.trainable = False
        
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.trainable = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages() 