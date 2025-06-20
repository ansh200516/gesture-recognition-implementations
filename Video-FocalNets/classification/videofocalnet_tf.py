# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
#
# TensorFlow/Keras implementation by Gemini
# --------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange

class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='gelu', drop=0., **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, name="fc1")
        self.act = layers.Activation(act_layer)
        self.fc2 = layers.Dense(out_features, name="fc2")
        self.drop = layers.Dropout(drop)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if self.drop_prob == 0. or not training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        random_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * random_tensor

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config

class DepthwiseConv1D(layers.Layer):
    def __init__(self, kernel_size, strides, padding, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        # We will build the conv layer in build()
        
    def build(self, input_shape):
        # input_shape is (batch, steps, channels)
        # We'll treat the 'steps' as height and '1' as width for a 2D conv
        # and channels become input channels for the 2D conv
        # The depthwise conv will have an input channel multiplier of 1
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=(self.kernel_size, 1),
            strides=(self.strides, 1),
            padding=self.padding,
            use_bias=self.use_bias,
            depth_multiplier=1
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, steps, channels)
        # expand to (batch, steps, 1, channels) for Conv2D
        x_expanded = tf.expand_dims(x, axis=2)
        # apply conv
        x_conv = self.depthwise_conv(x_expanded)
        # squeeze back to (batch, steps, channels)
        x_squeezed = tf.squeeze(x_conv, axis=2)
        return x_squeezed

class SpatioTemporalFocalModulation(layers.Layer):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False, num_frames=8, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.num_frames = num_frames

        self.f = layers.Dense(2 * dim + (self.focal_level + 1), use_bias=bias, name="f")
        self.h = layers.Conv2D(dim, kernel_size=1, strides=1, use_bias=bias, name="h")

        self.act = layers.Activation('gelu')
        self.proj = layers.Dense(dim, name="proj")
        self.proj_drop = layers.Dropout(proj_drop)
        self.focal_layers = []
        self.focal_layers_temporal = []

        self.f_temporal = layers.Dense(dim + (self.focal_level + 1), use_bias=bias, name="f_temporal")
        self.h_temporal = layers.Conv1D(dim, kernel_size=1, strides=1, use_bias=bias, name="h_temporal")

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                tf.keras.Sequential([
                    layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1,
                                           padding='same', use_bias=False,
                                           name=f"focal_layers/{k}/0"),
                    layers.Activation('gelu', name=f"focal_layers/{k}/1"),
                ])
            )

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers_temporal.append(
                tf.keras.Sequential([
                    DepthwiseConv1D(kernel_size=kernel_size, strides=1, padding='same', use_bias=False, name=f"focal_layers_temporal/{k}/0"),
                    layers.Activation('gelu', name=f"focal_layers_temporal/{k}/1"),
                ])
            )

        if self.use_postln_in_modulation:
            self.ln = layers.LayerNormalization(epsilon=1e-5, name="ln")

    def call(self, x, training=False):
        B_t, H, W, C = x.shape

        # Pre linear projection temporal
        x_temporal = tf.identity(x)
        x_temporal = rearrange(x_temporal, '(b t) h w c -> (b h w) t c', t=self.num_frames, h=H, w=W)
        x_temporal = self.f_temporal(x_temporal)
        ctx_temporal, gates_temporal = tf.split(x_temporal, [C, self.focal_level + 1], axis=-1)
        
        ctx_temporal = tf.transpose(ctx_temporal, perm=[0, 2, 1])
        gates_temporal = tf.transpose(gates_temporal, perm=[0, 2, 1])


        # Context aggregation temporal
        ctx_all_temporal = 0
        for l in range(self.focal_level):
            ctx_temporal = self.focal_layers_temporal[l](ctx_temporal)
            ctx_all_temporal = ctx_all_temporal + ctx_temporal * gates_temporal[:, l:l+1]
        ctx_global_temporal = self.act(tf.reduce_mean(ctx_temporal, axis=2, keepdims=True))
        ctx_all_temporal = ctx_all_temporal + ctx_global_temporal * gates_temporal[:, self.focal_level:]

        # Pre linear projection spatial
        x = self.f(x)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        q, ctx, gates = tf.split(x, [C, C, self.focal_level + 1], axis=1)

        # Context aggregation spatial
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l+1]
        ctx_global = self.act(tf.reduce_mean(ctx, axis=[2, 3], keepdims=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # Normalize context
        if self.normalize_modulator:
            ctx_all_temporal = ctx_all_temporal / (self.focal_level + 1)
            ctx_all = ctx_all / (self.focal_level + 1)

        # Focal modulation
        modulator_temporal = self.h_temporal(ctx_all_temporal)
        modulator_temporal = rearrange(modulator_temporal, '(b h w) c t -> (b t) c h w', t=self.num_frames, h=H, w=W)

        modulator = self.h(ctx_all)

        x_out = q * modulator * modulator_temporal
        x_out = tf.transpose(x_out, perm=[0, 2, 3, 1])
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # Post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out, training=training)
        return x_out


class VideoFocalNetBlock(layers.Layer):
    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer='gelu', norm_layer=layers.LayerNormalization,
                 focal_level=1, focal_window=3,
                 use_layerscale=False, layerscale_value=1e-4,
                 use_postln=False, use_postln_in_modulation=False,
                 normalize_modulator=False, num_frames=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        self.use_postln = use_postln

        self.norm1 = norm_layer(epsilon=1e-5, name="norm1")
        self.modulation = SpatioTemporalFocalModulation(
            dim, proj_drop=drop, focal_window=focal_window, focal_level=focal_level,
            use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator,
            num_frames=self.num_frames, name="modulation"
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else layers.Activation('linear')
        self.norm2 = norm_layer(epsilon=1e-5, name="norm2")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, name="mlp")

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = self.add_weight(shape=(dim,), initializer=tf.constant_initializer(layerscale_value), trainable=True, name="gamma_1")
            self.gamma_2 = self.add_weight(shape=(dim,), initializer=tf.constant_initializer(layerscale_value), trainable=True, name="gamma_2")

        self.H = None
        self.W = None

    def call(self, x, training=False):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation
        x_norm = x if self.use_postln else self.norm1(x)
        x_reshaped = tf.reshape(x_norm, (B, H, W, C))
        x_mod = self.modulation(x_reshaped, training=training)
        x_mod = tf.reshape(x_mod, (B, H * W, C))
        x_mod = x_mod if not self.use_postln else self.norm1(x_mod)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x_mod, training=training)
        x_ffn = self.norm2(self.mlp(x, training=training)) if self.use_postln else self.mlp(self.norm2(x), training=training)
        x = x + self.drop_path(self.gamma_2 * x_ffn, training=training)

        return x


class BasicLayer(layers.Layer):
    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=layers.LayerNormalization,
                 downsample=None, use_checkpoint=False,
                 focal_level=1, focal_window=1,
                 use_conv_embed=False,
                 use_layerscale=False, layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 num_frames=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_frames = num_frames

        self.blocks = [
            VideoFocalNetBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
                num_frames=self.num_frames,
                name=f"blocks/{i}"
            )
            for i in range(depth)]

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution,
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False,
                name='downsample'
            )
        else:
            self.downsample = None

    def call(self, x, H, W, training=False):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, training=training)

        if self.downsample is not None:
            B, L, C = x.shape
            x_reshaped = tf.reshape(tf.transpose(x, perm=[0, 2, 1]), (B, C, H, W))
            x_reshaped = tf.transpose(x_reshaped, perm=[0,2,3,1]) # B, H, W, C
            x, Ho, Wo = self.downsample(x_reshaped, training=training)
        else:
            Ho, Wo = H, W
        return x, Ho, Wo


class PatchEmbed(layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96,
                 use_conv_embed=False, norm_layer=None, is_stem=False, tubelet_size=1, **kwargs):
        super().__init__(**kwargs)
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size

        if use_conv_embed:
            if is_stem:
                kernel_size, padding, stride = 7, 2, 4
            else:
                kernel_size, padding, stride = 3, 1, 2
            self.proj = layers.Conv2D(embed_dim, kernel_size=kernel_size, strides=stride, padding='same' if padding > 0 else 'valid', name="proj")
        else:
            if self.tubelet_size == 1:
                self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, name="proj")
            else:
                self.proj = layers.Conv3D(embed_dim,
                                          kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                                          strides=(tubelet_size, patch_size[0], patch_size[1]), name="proj")

        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name="norm")
        else:
            self.norm = None

    def call(self, x, training=False):
        if self.tubelet_size == 1:
            x = self.proj(x)
            H, W = tf.shape(x)[1], tf.shape(x)[2]
            x = tf.reshape(x, (tf.shape(x)[0], H * W, self.embed_dim))
            if self.norm is not None:
                x = self.norm(x)
            return x, H, W
        else:
            x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
            x = self.proj(x)
            B, C, T, H, W = x.shape
            x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
            x = tf.reshape(x, (B * T, C, H, W))
            x = tf.transpose(x, perm=[0, 2, 3, 1])
            x_reshaped = tf.reshape(x, (tf.shape(x)[0], H * W, self.embed_dim))
            if self.norm is not None:
                x_reshaped = self.norm(x_reshaped)
            return x_reshaped, H, W


class VideoFocalNet(tf.keras.Model):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=layers.LayerNormalization,
                 patch_norm=True,
                 use_checkpoint=False,
                 focal_levels=[2, 2, 2, 2],
                 focal_windows=[3, 3, 3, 3],
                 use_conv_embed=False,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 num_frames=8,
                 tubelet_size=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_layers = len(depths)
        embed_dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.num_classes = num_classes
        self.embed_dim = embed_dims
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratio = mlp_ratio
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames // self.tubelet_size

        self.patch_embed = PatchEmbed(
            img_size=(img_size, img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            use_conv_embed=use_conv_embed,
            norm_layer=norm_layer if self.patch_norm else None,
            is_stem=True,
            tubelet_size=tubelet_size,
            name="patch_embed"
        )

        self.pos_drop = layers.Dropout(drop_rate)

        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

        self.layers_list = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                input_resolution=(self.patch_embed.patches_resolution[0] // (2 ** i_layer),
                                  self.patch_embed.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_conv_embed=use_conv_embed,
                use_checkpoint=use_checkpoint,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
                num_frames=self.num_frames,
                name=f"layers/{i_layer}"
            )
            self.layers_list.append(layer)

        self.norm = norm_layer(epsilon=1e-5, name="norm")
        self.avgpool = layers.GlobalAveragePooling1D()
        self.head = layers.Dense(num_classes, name="head") if num_classes > 0 else layers.Activation('linear')

    def forward_features(self, x, training=False):
        x, H, W = self.patch_embed(x, training=training)
        x = self.pos_drop(x, training=training)

        for layer in self.layers_list:
            x, H, W = layer(x, H, W, training=training)
        
        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x, training=False):
        b, t, c, h, w = x.shape
        
        # Pytorch input is (B, T, C, H, W).
        # TF Keras layers with 'channels_last' expect (B, H, W, C).
        # We need to reshape and transpose.
        
        if self.tubelet_size==1:
            # Reshape from (b, t, c, h, w) to (b*t, c, h, w)
            x =  tf.reshape(x, (-1, c, h, w))
            # Transpose to (b*t, h, w, c) for channels_last
            x = tf.transpose(x, perm=[0, 2, 3, 1])

        # for tubelet_size > 1, patch_embed handles the transform
        
        x = self.forward_features(x, training=training)
        x = tf.reshape(x, (b, self.num_frames, -1))
        x = tf.reduce_mean(x, axis=1)
        x = self.head(x)
        return x

def videofocalnet_tiny(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    return model

def videofocalnet_small(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    return model

def videofocalnet_base(pretrained=False, **kwargs):
    model = VideoFocalNet(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    return model

if __name__ == '__main__':
    print('TensorFlow VideoFocalNet')
    model = videofocalnet_tiny(num_classes=10)
    dummy_input = tf.random.uniform((1, 8, 3, 224, 224))
    output = model(dummy_input)
    print(output.shape) 