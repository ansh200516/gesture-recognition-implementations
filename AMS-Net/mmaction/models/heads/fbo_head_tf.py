import copy
import tensorflow as tf
from ..common.lfb_tf import LFB

class NonLocalLayer(tf.keras.layers.Layer):
    def __init__(self,
                 st_feat_channels,
                 lt_feat_channels,
                 latent_channels,
                 num_st_feat,
                 num_lt_feat,
                 use_scale=True,
                 pre_activate=True,
                 pre_activate_with_ln=True,
                 dropout_ratio=0.2,
                 zero_init_out_conv=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.use_scale = use_scale
        self.pre_activate = pre_activate
        self.pre_activate_with_ln = pre_activate_with_ln
        self.dropout_ratio = dropout_ratio
        self.zero_init_out_conv = zero_init_out_conv

        self.st_feat_conv = tf.keras.layers.Conv3D(
            self.latent_channels, kernel_size=1, name='st_feat_conv')
        self.lt_feat_conv = tf.keras.layers.Conv3D(
            self.latent_channels, kernel_size=1, name='lt_feat_conv')
        self.global_conv = tf.keras.layers.Conv3D(
            self.latent_channels, kernel_size=1, name='global_conv')

        if pre_activate:
            self.ln = tf.keras.layers.LayerNormalization(axis=[1, 2, 3, 4])
        else:
            self.ln = tf.keras.layers.LayerNormalization(axis=[1, 2, 3, 4])

        self.relu = tf.keras.layers.ReLU()

        out_conv_initializer = 'zeros' if self.zero_init_out_conv else 'he_normal'
        self.out_conv = tf.keras.layers.Conv3D(
            self.st_feat_channels, kernel_size=1, kernel_initializer=out_conv_initializer, name='out_conv')

        if self.dropout_ratio > 0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_ratio)

    def call(self, inputs, training=None):
        st_feat, lt_feat = inputs
        n = tf.shape(st_feat)[0]
        c = self.latent_channels
        num_st_feat, num_lt_feat = self.num_st_feat, self.num_lt_feat

        theta = self.st_feat_conv(st_feat)
        theta = tf.reshape(theta, (n, c, num_st_feat))

        phi = self.lt_feat_conv(lt_feat)
        phi = tf.reshape(phi, (n, c, num_lt_feat))

        g = self.global_conv(lt_feat)
        g = tf.reshape(g, (n, c, num_lt_feat))

        theta_phi = tf.linalg.matmul(tf.transpose(theta, perm=[0, 2, 1]), phi)
        if self.use_scale:
            theta_phi /= tf.sqrt(tf.cast(c, tf.float32))

        p = tf.nn.softmax(theta_phi, axis=-1)

        out = tf.linalg.matmul(g, tf.transpose(p, perm=[0, 2, 1]))
        out = tf.reshape(out, (n, c, num_st_feat, 1, 1))

        if self.pre_activate:
            if self.pre_activate_with_ln:
                out = self.ln(out)
            out = self.relu(out)

        out = self.out_conv(out)

        if not self.pre_activate:
            out = self.ln(out)
        if self.dropout_ratio > 0:
            out = self.dropout(out, training=training)

        return out


class FBONonLocal(tf.keras.layers.Layer):
    def __init__(self,
                 st_feat_channels,
                 lt_feat_channels,
                 latent_channels,
                 num_st_feat,
                 num_lt_feat,
                 num_non_local_layers=2,
                 st_feat_dropout_ratio=0.2,
                 lt_feat_dropout_ratio=0.2,
                 pre_activate=True,
                 zero_init_out_conv=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert num_non_local_layers >= 1, 'At least one non_local_layer is needed.'
        self.latent_channels = latent_channels
        self.num_non_local_layers = num_non_local_layers
        self.st_feat_dropout_ratio = st_feat_dropout_ratio
        self.lt_feat_dropout_ratio = lt_feat_dropout_ratio
        self.pre_activate = pre_activate

        self.st_feat_conv = tf.keras.layers.Conv3D(
            latent_channels, kernel_size=1, name='st_feat_conv')
        self.lt_feat_conv = tf.keras.layers.Conv3D(
            lt_feat_channels, kernel_size=1, name='lt_feat_conv')

        if self.st_feat_dropout_ratio > 0:
            self.st_feat_dropout = tf.keras.layers.Dropout(self.st_feat_dropout_ratio)

        if self.lt_feat_dropout_ratio > 0:
            self.lt_feat_dropout = tf.keras.layers.Dropout(self.lt_feat_dropout_ratio)

        if not self.pre_activate:
            self.relu = tf.keras.layers.ReLU()

        self.non_local_layers = [
            NonLocalLayer(
                latent_channels,
                latent_channels,
                latent_channels,
                num_st_feat,
                num_lt_feat,
                pre_activate=self.pre_activate,
                zero_init_out_conv=zero_init_out_conv,
                name=f'non_local_layer_{idx + 1}')
            for idx in range(self.num_non_local_layers)
        ]

    def call(self, inputs, training=None):
        st_feat, lt_feat = inputs
        st_feat = self.st_feat_conv(st_feat)
        if self.st_feat_dropout_ratio > 0:
            st_feat = self.st_feat_dropout(st_feat, training=training)

        lt_feat = self.lt_feat_conv(lt_feat)
        if self.lt_feat_dropout_ratio > 0:
            lt_feat = self.lt_feat_dropout(lt_feat, training=training)

        for non_local_layer in self.non_local_layers:
            identity = st_feat
            nl_out = non_local_layer([st_feat, lt_feat], training=training)
            nl_out = identity + nl_out
            if not self.pre_activate:
                nl_out = self.relu(nl_out)
            st_feat = nl_out

        return nl_out


class FBOAvg(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        st_feat, lt_feat = inputs
        # NCTHW format, pool over T
        out = tf.reduce_mean(lt_feat, axis=2, keepdims=True)
        return out


class FBOMax(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        st_feat, lt_feat = inputs
        # NCTHW format, pool over T
        out = tf.reduce_max(lt_feat, axis=2, keepdims=True)
        return out


class FBOHead(tf.keras.layers.Layer):
    fbo_dict = {'non_local': FBONonLocal, 'avg': FBOAvg, 'max': FBOMax}

    def __init__(self,
                 lfb_cfg,
                 fbo_cfg,
                 temporal_pool_type='avg',
                 spatial_pool_type='max',
                 **kwargs):
        super().__init__(**kwargs)
        fbo_type = fbo_cfg.pop('type', 'non_local')
        assert fbo_type in FBOHead.fbo_dict
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']

        self.lfb_cfg = copy.deepcopy(lfb_cfg)
        self.fbo_cfg = copy.deepcopy(fbo_cfg)

        self.lfb = LFB(**self.lfb_cfg)
        self.fbo = self.fbo_dict[fbo_type](**self.fbo_cfg)

        if temporal_pool_type == 'avg':
            self.temporal_pool = lambda x: tf.reduce_mean(x, axis=2, keepdims=True)
        else:
            self.temporal_pool = lambda x: tf.reduce_max(x, axis=2, keepdims=True)
        
        if spatial_pool_type == 'avg':
            self.spatial_pool = lambda x: tf.reduce_mean(x, axis=[3, 4], keepdims=True)
        else:
            self.spatial_pool = lambda x: tf.reduce_max(x, axis=[3, 4], keepdims=True)

    def sample_lfb(self, rois, lfb_data):
        """Sample long-term features for each ROI feature."""
        # In TF, we assume lfb_data is already prepared and passed to call().
        # This function might need adjustment depending on how lfb_data is structured and passed.
        # The original PyTorch version iterates through img_metas, which is not TF-Graph friendly.
        # Here we call the LFB layer.
        lt_feat = self.lfb(lfb_data)
        return lt_feat

    def call(self, inputs, training=None):
        x, rois, lfb_data = inputs
        
        st_feat = self.temporal_pool(x)
        st_feat = self.spatial_pool(st_feat)
        identity = st_feat

        lt_feat = self.sample_lfb(rois, lfb_data)
        
        fbo_feat = self.fbo([st_feat, lt_feat], training=training)

        out = tf.concat([identity, fbo_feat], axis=1)
        return out 