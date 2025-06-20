import logging
import os
from collections import OrderedDict
from functools import partial
from itertools import repeat
import collections.abc

import numpy as np
import scipy
import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange
from tensorflow.keras import Model, layers

# Model Registry
_model_entrypoints = {}

def register_model(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]
    _model_entrypoints[model_name] = fn
    return fn

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class TruncatedNormal(tf.keras.initializers.Initializer):
    def __init__(self, stddev=0.02):
        super().__init__()
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        return tf.random.truncated_normal(shape, mean=0.0, stddev=self.stddev, dtype=dtype)

    def get_config(self):
        return {'stddev': self.stddev}


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
    random_tensor = tf.floor(random_tensor)
    output = tf.math.divide(x, keep_prob) * random_tensor
    return output


class DropPath(layers.Layer):
    def __init__(self, drop_prob=0., **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)

    def get_config(self):
        config = super().get_config()
        config.update({'drop_prob': self.drop_prob})
        return config


class QuickGELU(layers.Layer):
    def call(self, x):
        return x * tf.sigmoid(1.702 * x)


class Mlp(layers.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=QuickGELU,
                 drop=0.,
                 **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, name='fc1')
        self.act = act_layer()
        self.fc2 = layers.Dense(out_features, name='fc2')
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(layers.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method, 'q'
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method, 'k'
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method, 'v'
        )

        self.proj_q = layers.Dense(dim_out, use_bias=qkv_bias, name='proj_q')
        self.proj_k = layers.Dense(dim_out, use_bias=qkv_bias, name='proj_k')
        self.proj_v = layers.Dense(dim_out, use_bias=qkv_bias, name='proj_v')

        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim_out, name='proj')
        self.proj_drop = layers.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method,
                          name_prefix):
        if method == 'dw_bn':
            proj = tf.keras.Sequential([
                layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same' if padding > 0 else 'valid',
                    use_bias=False,
                    name=f'conv_proj_{name_prefix}/conv'
                ),
                layers.BatchNormalization(name=f'conv_proj_{name_prefix}/bn'),
                Rearrange('b h w c -> b (h w) c'),
            ])
        elif method == 'avg':
            proj = tf.keras.Sequential([
                layers.AvgPool2D(
                    pool_size=kernel_size,
                    strides=stride,
                    padding='same' if padding > 0 else 'valid',
                ),
                Rearrange('b h w c -> b (h w) c'),
            ])
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = tf.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b h w c', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b h w c -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b h w c -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b h w c -> b (h w) c')

        if self.with_cls_token:
            q = tf.concat([cls_token, q], axis=1)
            k = tf.concat([cls_token, k], axis=1)
            v = tf.concat([cls_token, v], axis=1)

        return q, k, v

    def call(self, x, h, w):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)
        else:
            q, k, v = x, x, x

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = tf.einsum('bhlk,bhtk->bhlt', q, k) * self.scale
        attn = tf.nn.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.einsum('bhlt,bhtv->bhlv', attn, v)
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(layers.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=QuickGELU,
                 norm_layer=layers.LayerNormalization,
                 **kwargs):
        super().__init__(**kwargs)

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(epsilon=1e-5, name='norm1')
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            name='attn',
            **kwargs
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else layers.Activation('linear')
        self.norm2 = norm_layer(epsilon=1e-5, name='norm2')

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop,
            name='mlp'
        )

    def call(self, x_input):
        x, h, w = x_input
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, h, w


class ConvEmbed(layers.Layer):
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = to_2tuple(patch_size)

        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding='same' if padding > 0 else 'valid',
            name='proj'
        )
        self.norm = norm_layer(epsilon=1e-5, name='norm') if norm_layer else None

    def call(self, x):
        x = self.proj(x)

        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)

        return x


class VisionTransformer(Model):
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=QuickGELU,
                 norm_layer=layers.LayerNormalization,
                 init='trunc_norm',
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            name='patch_embed'
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = self.add_weight(
                name='cls_token',
                shape=(1, 1, embed_dim),
                initializer=tf.zeros_initializer(),
                trainable=True
            )
        else:
            self.cls_token = None

        self.pos_drop = layers.Dropout(p=drop_rate)
        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]

        self.blocks = []
        for j in range(depth):
            self.blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    name=f'blocks/{j}',
                    **kwargs
                )
            )

        if self.cls_token is not None:
            initializer = TruncatedNormal(stddev=.02)
            self.cls_token.assign(initializer(shape=self.cls_token.shape))

        # Weight init is handled by layer initializers.
        # The `init` parameter is kept for API consistency with original model.
        # Keras layers have sane defaults. `Dense` uses 'glorot_uniform'.

    def call(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w c -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = tf.tile(self.cls_token, [B, 1, 1])
            x = tf.concat([cls_tokens, x], axis=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x, H, W = blk((x, H, W))

        if self.cls_token is not None:
            cls_tokens, x = tf.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)

        return x, cls_tokens


class ConvolutionalVisionTransformer(Model):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=QuickGELU,
                 norm_layer=layers.LayerNormalization,
                 init='trunc_norm',
                 spec=None,
                 **kwargs):
        super().__init__(name='ConvolutionalVisionTransformer', **kwargs)
        self.num_classes = num_classes
        self.num_stages = spec['NUM_STAGES']
        self.stages = []

        for i in range(self.num_stages):
            stage_kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                name=f'stage{i}',
                **stage_kwargs
            )
            self.stages.append(stage)
            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.cls_token_final = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = layers.Dense(num_classes, name='head', kernel_initializer=TruncatedNormal(stddev=0.02)) if num_classes > 0 else layers.Activation('linear')

    def forward_features(self, x):
        cls_tokens = None
        for stage in self.stages:
            x, cls_tokens = stage(x)

        if self.cls_token_final:
            x = self.norm(cls_tokens)
            x = tf.squeeze(x, axis=1)
        else:
            x = rearrange(x, 'b h w c -> b (h w) c')
            x = self.norm(x)
            x = tf.reduce_mean(x, axis=1)

        return x

    def call(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def get_cls_model(config, **kwargs):
    msvit_spec = config.MODEL.SPEC
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        # Weight loading from PyTorch checkpoints requires a manual process
        # of mapping weights, which is not implemented here.
        # This would typically be replaced with TF checkpoint loading.
        logging.info("TensorFlow model created, weight initialization from PyTorch checkpoint not supported out-of-the-box.")
        logging.info(f"Pretrained path: {config.MODEL.PRETRAINED}")
        
    return msvit 