import tensorflow as tf
import numpy as np
from einops import rearrange
from functools import reduce, lru_cache
from operator import mul


class Mlp(tf.keras.layers.Layer):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.keras.layers.GELU, drop=0., **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Dense(hidden_features, name='fc1')
        self.act = act_layer()
        self.fc2 = tf.keras.layers.Dense(out_features, name='fc2')
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, prod(window_size), C)
    """
    B, D, H, W, C = x.shape
    x = tf.reshape(x, [B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C])
    windows = tf.transpose(x, perm=[0, 1, 3, 5, 2, 4, 6, 7])
    windows = tf.reshape(windows, [-1, np.prod(window_size), C])
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, prod(window_size), C)
        window_size (tuple[int]): Window size
        B, D, H, W (int): B, D, H, W of original feature map

    Returns:
        x: (B, D, H, W, C)
    """
    C = windows.shape[-1]
    x = tf.reshape(windows, [B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], C])
    x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])
    x = tf.reshape(x, [B, D, H, W, C])
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0., **kwargs):
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


class WindowAttention3D(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias, name='qkv')
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim, name='proj')
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            'relative_position_bias_table',
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype)

        coords_d = tf.range(self.window_size[0])
        coords_h = tf.range(self.window_size[1])
        coords_w = tf.range(self.window_size[2])
        coords = tf.stack(tf.meshgrid(coords_d, coords_h, coords_w, indexing='ij'), axis=0)
        coords_flatten = tf.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        
        relative_coords_0 = relative_coords[..., 0] + self.window_size[0] - 1
        relative_coords_1 = relative_coords[..., 1] + self.window_size[1] - 1
        relative_coords_2 = relative_coords[..., 2] + self.window_size[2] - 1
        
        relative_coords_0 *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords_1 *= (2 * self.window_size[2] - 1)
        
        self.relative_position_index = tf.cast(relative_coords_0 + relative_coords_1 + relative_coords_2, dtype=tf.int32)
        super().build(input_shape)

    def call(self, x, mask=None, training=False):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)

        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index[:N, :N], [-1]))
        relative_position_bias = tf.reshape(relative_position_bias, [N, N, -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + relative_position_bias[tf.newaxis, ...]

        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + mask[tf.newaxis, :, tf.newaxis, :, :]
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, C])
        
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class SwinTransformerBlock3D(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.layers.GELU, norm_layer=tf.keras.layers.LayerNormalization, use_checkpoint=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5, name='norm1')
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, name='attn')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.keras.layers.Activation('linear')
        self.norm2 = norm_layer(epsilon=1e-5, name='norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, name='mlp')

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_h0 = 0
        pad_h1 = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w0 = 0
        pad_w1 = (window_size[2] - W % window_size[2]) % window_size[2]
        
        x = tf.pad(x, [[0, 0], [pad_d0, pad_d1], [pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]])
        _, Dp, Hp, Wp, _ = x.shape
        
        if any(i > 0 for i in shift_size):
            shifted_x = tf.roll(x, shift=[-shift_size[0], -shift_size[1], -shift_size[2]], axis=[1, 2, 3])
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        
        if any(i > 0 for i in shift_size):
            x = tf.roll(shifted_x, shift=[shift_size[0], shift_size[1], shift_size[2]], axis=[1, 2, 3])
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0:
            x = x[:, :D, :H, :W, :]
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def call(self, x, mask_matrix, training=False):
        shortcut = x
        
        if self.use_checkpoint:
            x = tf.recompute_grad(self.forward_part1)(x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        
        x = shortcut + self.drop_path(x, training=training)

        if self.use_checkpoint:
            x_mlp = tf.recompute_grad(self.forward_part2)(x)
        else:
            x_mlp = self.forward_part2(x)
        
        x = x + x_mlp
        
        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, dim, norm_layer=tf.keras.layers.LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False, name='reduction')
        self.norm = norm_layer(epsilon=1e-5, name='norm')

    def call(self, x):
        B, D, H, W, C = x.shape

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            paddings = [[0, 0], [0, 0], [0, H % 2], [0, W % 2], [0, 0]]
            x = tf.pad(x, paddings)

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    with tf.device(device):
        img_mask = tf.zeros((1, D, H, W, 1), dtype=tf.int32)
        cnt = 0
        slices = (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None))
        h_slices = (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None))
        w_slices = (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None))
        
        for d in slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[0, d, h, w, 0].assign(cnt)
                    cnt += 1
        
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.squeeze(mask_windows, axis=-1)
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, -100.0, 0.0)
    return attn_mask


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, depth, num_heads, window_size=(1,7,7), mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=tf.keras.layers.LayerNormalization, downsample=None, use_checkpoint=False, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = [
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                name=f'blocks.{i}'
            )
            for i in range(depth)]
        
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, name='downsample')
        else:
            self.downsample = None

    def call(self, x, training=False):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        
        for blk in self.blocks:
            x = blk(x, attn_mask, training=training)
        
        x = tf.reshape(x, [B, D, H, W, -1])

        if self.downsample is not None:
            x = self.downsample(x)
        
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(tf.keras.layers.Layer):
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = tf.keras.layers.Conv3D(embed_dim, kernel_size=patch_size, strides=patch_size, name='proj', data_format='channels_first')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        _, _, D, H, W = x.shape
        if W % self.patch_size[2] != 0:
            pad_w = self.patch_size[2] - W % self.patch_size[2]
            paddings = tf.constant([[0,0], [0,0], [0,0], [0,0], [0, pad_w]])
            x = tf.pad(x, paddings)
        if H % self.patch_size[1] != 0:
            pad_h = self.patch_size[1] - H % self.patch_size[1]
            paddings = tf.constant([[0,0], [0,0], [0,0], [0, pad_h], [0,0]])
            x = tf.pad(x, paddings)
        if D % self.patch_size[0] != 0:
            pad_d = self.patch_size[0] - D % self.patch_size[0]
            paddings = tf.constant([[0,0], [0,0], [0, pad_d], [0,0], [0,0]])
            x = tf.pad(x, paddings)

        x = self.proj(x)
        if self.norm is not None:
            D, H, W = x.shape[2], x.shape[3], x.shape[4]
            x = tf.reshape(x, [tf.shape(x)[0], self.embed_dim, -1])
            x = tf.transpose(x, perm=[0, 2, 1])
            x = self.norm(x)
            x = tf.transpose(x, perm=[0, 2, 1])
            x = tf.reshape(x, [-1, self.embed_dim, D, H, W])
        return x


class SwinTransformer3D(tf.keras.Model):
    def __init__(self,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=tf.keras.layers.LayerNormalization,
                 patch_norm=False,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, name='patch_embed')

        self.pos_drop = tf.keras.layers.Dropout(drop_rate)

        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        self.layers_list = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                name=f'layers.{i_layer}')
            self.layers_list.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))
        self.norm = norm_layer(epsilon=1e-5, name='norm')

    def call(self, x, training=False):
        x = self.patch_embed(x)
        x = self.pos_drop(x, training=training)

        for layer in self.layers_list:
            x = layer(x, training=training)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x 