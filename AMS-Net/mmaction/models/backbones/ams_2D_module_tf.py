import tensorflow as tf
from tensorflow.keras import layers


class Temporal_Block(layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super(Temporal_Block, self).__init__(**kwargs)
        self.channels = channels

        # temporal
        self.conv = layers.Conv3D(
            self.channels,
            kernel_size=(3, 1, 1),
            strides=1,
            padding='same',
            dilation_rate=(1, 1, 1),
            use_bias=False,
            data_format='channels_first'
        )
        self.bn = layers.BatchNormalization(axis=1)  # axis=1 for channels_first
        self.relu = layers.ReLU()

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class MultiScale_Temporal_Block(layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super(MultiScale_Temporal_Block, self).__init__(**kwargs)
        self.channels = channels

        self.Temporal_Block_1 = Temporal_Block(self.channels)
        self.Temporal_Block_2 = Temporal_Block(self.channels)
        self.Temporal_Block_3 = Temporal_Block(self.channels)

    def call(self, x, training=None):
        # temporal
        output1 = self.Temporal_Block_1(x, training=training)
        output2 = self.Temporal_Block_2(output1, training=training)
        output3 = self.Temporal_Block_3(output2, training=training)
        output = (output1 + output2 + output3) / 3.0
        return output


class MultiScale_Temporal(layers.Layer):
    def __init__(self, channels: int, segments: int, **kwargs):
        super(MultiScale_Temporal, self).__init__(**kwargs)
        self.channels = channels
        self.segments = segments
        self.temporal = MultiScale_Temporal_Block(self.channels)

    def call(self, x, training=None):
        # Input shape: (N*S, C, H, W)
        shape = tf.shape(x)
        c, h, w = shape[1], shape[2], shape[3]

        # Reshape from (N*S, C, H, W) to (N, S, C, H, W)
        x_reshaped = tf.reshape(x, (-1, self.segments, c, h, w))

        # Transpose to (N, C, S, H, W) for 3D convolution over T=S
        x_transposed = tf.transpose(x_reshaped, perm=[0, 2, 1, 3, 4])

        # temporal
        output = self.temporal(x_transposed, training=training)

        # Transpose back to (N, S, C, H, W)
        output_transposed = tf.transpose(output, perm=[0, 2, 1, 3, 4])

        # Reshape to (N*S, C, H, W)
        output_reshaped = tf.reshape(output_transposed, (-1, c, h, w))

        return output_reshaped


class Competitive_Progressive_Temporal_Module(layers.Layer):
    def __init__(self, opt_block, inplance, branchs, rate, stride=1, L=32, gamma=1, **kwargs):
        super(Competitive_Progressive_Temporal_Module, self).__init__(**kwargs)
        d = max(int(inplance / rate), L)
        self.branchs = branchs
        self.inplance = inplance
        self.gamma = gamma
        print("CompetitiveFusion", self.gamma)

        self.temporal_blocks = [opt_block(self.inplance) for _ in range(self.branchs)]

        self.avgpool = layers.GlobalAveragePooling3D(data_format='channels_first')
        self.fc = layers.Dense(d, use_bias=False)
        self.bn = layers.BatchNormalization(axis=-1)
        self.relu = layers.ReLU()

        self.fcs = [layers.Dense(self.inplance) for _ in range(self.branchs)]
        self.softmax = layers.Softmax(axis=1)

    def call(self, x, training=None):
        # Input x shape: (N, C, T, H, W)
        pro_feas_list = []
        fea = x
        for i, temporal_block in enumerate(self.temporal_blocks):
            fea = temporal_block(fea, training=training)
            pro_feas_list.append(tf.expand_dims(fea, axis=1))

        pro_feas = tf.concat(pro_feas_list, axis=1)

        fea_U = tf.reduce_sum(pro_feas, axis=1)
        fea_U = fea_U / self.branchs
        fea_s = self.avgpool(fea_U)

        fea_h = self.fc(fea_s)
        fea_h = self.bn(fea_h, training=training)
        fea_h = self.relu(fea_h)

        attention_vectors_list = []
        for i, fc in enumerate(self.fcs):
            vector = tf.expand_dims(fc(fea_h), axis=1)
            attention_vectors_list.append(vector)
        attention_vectors = tf.concat(attention_vectors_list, axis=1)

        attention_vectors *= self.gamma
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = tf.expand_dims(tf.expand_dims(tf.expand_dims(attention_vectors, -1), -1), -1)

        # pro_feas shape: [N, branchs, C, T, H, W]
        # attention_vectors shape: [N, branchs, C, 1, 1, 1]
        fea_v = tf.reduce_sum(pro_feas * attention_vectors, axis=1)
        
        # fea_v shape: [N, C, T, H, W]
        # Reshape to 2D for restore conv
        shape = tf.shape(fea_v)
        n, c, t, h, w = shape[0], shape[1], shape[2], shape[3], shape[4]
        fea_v = tf.transpose(fea_v, perm=[0, 2, 1, 3, 4])
        fea_v = tf.reshape(fea_v, (n * t, c, h, w))
        
        return fea_v


class CSTP_Stage1_Adaptive_Fusion(layers.Layer):
    def __init__(self, inplance, branchs, rate, stride=1, L=32, gamma=1, **kwargs):
        super(CSTP_Stage1_Adaptive_Fusion, self).__init__(**kwargs)
        d = max(int(inplance / rate), L)
        self.branchs = branchs
        self.inplance = inplance
        self.gamma = gamma
        print("CSTP_stage1", self.gamma)

        self.avgpool = layers.GlobalAveragePooling3D(data_format='channels_first')
        self.fc = layers.Dense(d, use_bias=False)
        self.bn = layers.BatchNormalization(axis=-1)
        self.relu = layers.ReLU()

        self.fcs = [layers.Dense(self.inplance) for _ in range(self.branchs)]
        self.softmax = layers.Softmax(axis=1)

    def call(self, x, training=None):
        # x is a list of tensors
        flow_feas_list = [tf.expand_dims(fea, axis=1) for fea in x]
        flow_feas = tf.concat(flow_feas_list, axis=1)

        fea_U = tf.reduce_sum(flow_feas, axis=1)
        fea_s = self.avgpool(fea_U)
        fea_z = self.fc(fea_s)
        fea_z = self.bn(fea_z, training=training)
        fea_z = self.relu(fea_z)

        attention_vectors_list = []
        for i, fc in enumerate(self.fcs):
            vector = tf.expand_dims(fc(fea_z), axis=1)
            attention_vectors_list.append(vector)
        attention_vectors = tf.concat(attention_vectors_list, axis=1)

        attention_vectors *= self.gamma
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = tf.expand_dims(tf.expand_dims(tf.expand_dims(attention_vectors, -1), -1), -1)
        
        fea_v = flow_feas * attention_vectors
        
        split_feas = tf.split(fea_v, num_or_size_splits=self.branchs, axis=1)
        top_down_x, top_down_y, bottom_up_x, bottom_up_y = split_feas

        top_down_fea = tf.concat([top_down_x, top_down_y], axis=2)
        top_down_fea = tf.squeeze(top_down_fea, axis=1)

        bottom_up_fea = tf.concat([bottom_up_x, bottom_up_y], axis=2)
        bottom_up_fea = tf.squeeze(bottom_up_fea, axis=1)

        return top_down_fea, bottom_up_fea


class CSTP_Stage2_Adaptive_Fusion(layers.Layer):
    def __init__(self, inplance, branchs, rate, stride=1, L=32, gamma=1, **kwargs):
        super(CSTP_Stage2_Adaptive_Fusion, self).__init__(**kwargs)
        d = max(int(inplance / rate), L)
        self.branchs = branchs
        self.inplance = inplance
        self.gamma = gamma
        print("CSTP_stage2", self.gamma)

        self.avgpool = layers.GlobalAveragePooling3D(data_format='channels_first')
        self.fc = layers.Dense(d, use_bias=False)
        self.bn = layers.BatchNormalization(axis=-1)
        self.relu = layers.ReLU()

        self.fcs = [layers.Dense(self.inplance) for _ in range(self.branchs)]
        self.softmax = layers.Softmax(axis=1)

    def call(self, x, training=None):
        flow_feas_list = [tf.expand_dims(fea, axis=1) for fea in x]
        flow_feas = tf.concat(flow_feas_list, axis=1)
        
        fea_U = tf.reduce_sum(flow_feas, axis=1)
        fea_s = self.avgpool(fea_U)
        fea_z = self.fc(fea_s)
        fea_z = self.bn(fea_z, training=training)
        fea_z = self.relu(fea_z)

        attention_vectors_list = []
        for i, fc in enumerate(self.fcs):
            vector = tf.expand_dims(fc(fea_z), axis=1)
            attention_vectors_list.append(vector)
        attention_vectors = tf.concat(attention_vectors_list, axis=1)

        attention_vectors *= self.gamma
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = tf.expand_dims(tf.expand_dims(tf.expand_dims(attention_vectors, -1), -1), -1)
        
        fea_v = flow_feas * attention_vectors
        
        split_feas = tf.split(fea_v, num_or_size_splits=self.branchs, axis=1)
        top_down_fea, bottom_up_fea = split_feas
        
        fea_v = tf.concat([top_down_fea, bottom_up_fea], axis=2)
        fea_v = tf.squeeze(fea_v, axis=1)

        return fea_v


class MultiScale_Temporal_Module(layers.Layer):
    def __init__(self, channels: int, segments=8, **kwargs):
        super(MultiScale_Temporal_Module, self).__init__(**kwargs)
        self.channels = channels
        self.segments = segments
        self.cptm = Competitive_Progressive_Temporal_Module(Temporal_Block, self.channels, branchs=3, rate=16)

    def call(self, x, training=None):
        # Input shape: (N*S, C, H, W)
        shape = tf.shape(x)
        c, h, w = shape[1], shape[2], shape[3]
        
        # Reshape to (N, S, C, H, W)
        x_reshaped = tf.reshape(x, (-1, self.segments, c, h, w))
        
        # Transpose to (N, C, S, H, W)
        x_transposed = tf.transpose(x_reshaped, perm=[0, 2, 1, 3, 4])
        
        # temporal
        output = self.cptm(x_transposed, training=training)
        
        return output


class CPTM_Bottleneck(layers.Layer):
    def __init__(self, channels: int, percent=0.25, gamma=1, segments=8, **kwargs):
        super(CPTM_Bottleneck, self).__init__(**kwargs)
        self.channels = channels
        self.percent = percent
        self.current_channels = int(self.channels * self.percent)
        self.gamma = gamma
        self.segments = segments

        self.reduce = layers.Conv2D(self.current_channels, kernel_size=1, strides=1, padding='same', use_bias=False, data_format='channels_first')
        self.bn_re = layers.BatchNormalization(axis=1)

        self.cptm_bottleneck = Competitive_Progressive_Temporal_Module(Temporal_Block, self.current_channels, branchs=3, rate=16, gamma=self.gamma)

        self.restore = layers.Conv2D(self.channels, kernel_size=1, strides=1, padding='same', use_bias=False, data_format='channels_first')
        self.bn_restore = layers.BatchNormalization(axis=1)
        self.relu = layers.ReLU()

    def call(self, x, training=None):
        # x shape: (N*S, C, H, W)
        x = self.reduce(x)
        x = self.bn_re(x, training=training)
        x = self.relu(x)

        # Reshape to 3D for CPTM
        shape = tf.shape(x)
        c, h, w = shape[1], shape[2], shape[3]
        x_reshaped = tf.reshape(x, (-1, self.segments, c, h, w))
        x_transposed = tf.transpose(x_reshaped, perm=[0, 2, 1, 3, 4])

        output = self.cptm_bottleneck(x_transposed, training=training)

        output = self.restore(output)
        output = self.bn_restore(output, training=training)
        output = self.relu(output)

        return output 