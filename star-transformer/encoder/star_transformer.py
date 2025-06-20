r"""
A TensorFlow implementation of the encoder part of the **Star-Transformer**.
"""

__all__ = [
    "StarTransformer"
]

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class StarTransformer(layers.Layer):
    r"""
    The encoder part of the **Star-Transformer**. It takes a 3D tensor as input and returns
    a sequence of embeddings of the same length.
    Based on the paper `Star-Transformer <https://arxiv.org/abs/1902.09113>`_.

    :param hidden_size: The size of the input and output dimensions.
    :param num_layers: The number of **Star-Transformer** layers.
    :param num_head: The number of heads in the **multi-head attention**, must be divisible by `d_model`.
    :param head_dim: The dimension of each `head`.
    :param dropout: The dropout probability.
    :param max_len: If it is an `int`, it represents the maximum length of the input sequence,
        and the model will add `position embedding` to the input sequence; if it is `None`, this
        step will be skipped.
    """

    def __init__(self, hidden_size: int, num_layers: int, num_head: int, head_dim: int, dropout: float = 0.1,
                 max_len: int = None, **kwargs):
        super(StarTransformer, self).__init__(**kwargs)
        self.iters = num_layers

        self.norm = [layers.LayerNormalization(epsilon=1e-6, name=f'norm_{i}') for i in range(self.iters)]
        self.emb_drop = layers.Dropout(dropout)
        self.ring_att = [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0, name=f'ring_att_{i}')
                         for i in range(self.iters)]
        self.star_att = [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0, name=f'star_att_{i}')
                         for i in range(self.iters)]

        if max_len is not None:
            self.pos_emb = layers.Embedding(max_len, hidden_size)
        else:
            self.pos_emb = None

    def call(self, data: tf.Tensor, mask: tf.Tensor, training=False):
        r"""
        :param data: Input sequence, with shape `[batch_size, length, hidden]`
        :param mask: Padding mask for the input sequence, with shape `[batch_size, length]`.
                     Padded positions are indicated by **0**.
        :param training: Whether the call is in training mode.
        :return: A tuple containing two elements:
                 - The first element has a shape of `[batch_size, length, hidden]` and represents
                   the encoded output sequence.
                 - The second element has a shape of `[batch_size, hidden]` and represents the
                   global relay node, as detailed in the paper.
        """

        def norm_func(f, x, training=False):
            # B, H, L, 1 -> B, L, 1, H -> apply norm -> B, H, L, 1
            x_permuted = tf.transpose(x, perm=[0, 2, 3, 1])
            normed = f(x_permuted, training=training)
            return tf.transpose(normed, perm=[0, 3, 1, 2])

        B, L, H = tf.unstack(tf.shape(data))
        mask_bool = tf.cast(mask, dtype=tf.bool)
        
        smask = tf.concat([tf.zeros((B, 1), dtype=tf.bool), mask_bool], 1)

        embs = tf.expand_dims(tf.transpose(data, perm=[0, 2, 1]), -1)  # B, H, L, 1
        if self.pos_emb:
            P = tf.range(L, dtype=tf.int32)
            P = self.pos_emb(P)  # L, H
            P = tf.expand_dims(tf.expand_dims(tf.transpose(P, perm=[1, 0]), 0), -1)  # 1, H, L, 1
            embs = embs + P

        embs = norm_func(self.emb_drop, embs, training=training)
        nodes = embs
        relay = tf.reduce_mean(embs, axis=2, keepdims=True)
        ex_mask = tf.expand_dims(tf.expand_dims(mask_bool, 1), 3) # B, 1, L, 1
        ex_mask = tf.tile(ex_mask, [1, H, 1, 1]) # B, H, L, 1
        
        r_embs = tf.reshape(embs, (B, H, 1, L))

        for i in range(self.iters):
            ax = tf.concat([r_embs, tf.tile(relay, [1, 1, 1, L])], 2)
            nodes = tf.nn.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes, training=training), ax=ax, training=training))
            relay = tf.nn.leaky_relu(self.star_att[i](relay, tf.concat([relay, nodes], 2), smask, training=training))

            nodes = tf.where(ex_mask, 0.0, nodes)

        nodes = tf.transpose(tf.squeeze(nodes, -1), perm=[0, 2, 1])
        return nodes, tf.squeeze(relay, axis=[-1, -2])


class _MSA1(layers.Layer):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1, **kwargs):
        super(_MSA1, self).__init__(**kwargs)
        self.WQ = layers.Conv2D(nhead * head_dim, 1, data_format='channels_first', name='wq')
        self.WK = layers.Conv2D(nhead * head_dim, 1, data_format='channels_first', name='wk')
        self.WV = layers.Conv2D(nhead * head_dim, 1, data_format='channels_first', name='wv')
        self.WO = layers.Conv2D(nhid, 1, data_format='channels_first', name='wo')
        self.drop = layers.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def call(self, x, ax=None, training=False):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = tf.unstack(tf.shape(x))

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = tf.shape(ax)[2]
            ak = self.WK(ax) # B, C, aL, L
            av = self.WV(ax) # B, C, aL, L
            ak = tf.reshape(ak, (B, nhead, head_dim, aL, L))
            av = tf.reshape(av, (B, nhead, head_dim, aL, L))

        q = tf.reshape(q, (B, nhead, head_dim, 1, L))

        # F.unfold in PyTorch -> tf.image.extract_patches
        k_unfolded = tf.transpose(k, [0, 2, 3, 1]) # B, L, 1, C
        k_unfolded = tf.image.extract_patches(k_unfolded, [1, unfold_size, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME') # B, L, 1, C*unfold_size
        k_unfolded = tf.reshape(k_unfolded, (B, L, 1, nhead*head_dim, unfold_size))
        k_unfolded = tf.transpose(k_unfolded, [0, 3, 4, 1, 2]) # B, C, unfold_size, L, 1
        k = tf.reshape(k_unfolded, (B, nhead, head_dim, unfold_size, L))
        
        v_unfolded = tf.transpose(v, [0, 2, 3, 1])
        v_unfolded = tf.image.extract_patches(v_unfolded, [1, unfold_size, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
        v_unfolded = tf.reshape(v_unfolded, (B, L, 1, nhead*head_dim, unfold_size))
        v_unfolded = tf.transpose(v_unfolded, [0, 3, 4, 1, 2])
        v = tf.reshape(v_unfolded, (B, nhead, head_dim, unfold_size, L))

        if ax is not None:
            k = tf.concat([k, ak], 3)
            v = tf.concat([v, av], 3)

        alphas = tf.reduce_sum(q * k, axis=2, keepdims=True) / tf.sqrt(tf.cast(head_dim, q.dtype))
        alphas = tf.nn.softmax(alphas, axis=3)
        alphas = self.drop(alphas, training=training)

        att = tf.reduce_sum(alphas * v, axis=3)
        att = tf.transpose(att, [0, 1, 3, 2]) # B, nhead, L, head_dim
        att = tf.reshape(att, (B, nhead * head_dim, L, 1))

        ret = self.WO(att)
        return ret


class _MSA2(layers.Layer):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1, **kwargs):
        super(_MSA2, self).__init__(**kwargs)
        self.WQ = layers.Conv2D(nhead * head_dim, 1, data_format='channels_first', name='wq')
        self.WK = layers.Conv2D(nhead * head_dim, 1, data_format='channels_first', name='wk')
        self.WV = layers.Conv2D(nhead * head_dim, 1, data_format='channels_first', name='wv')
        self.WO = layers.Conv2D(nhid, 1, data_format='channels_first', name='wo')
        self.drop = layers.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim = nhid, nhead, head_dim

    def call(self, x, y, mask=None, training=False):
        # x: B, H, 1, 1,  y: B H L 1
        nhid, nhead, head_dim = self.nhid, self.nhead, self.head_dim
        B, H, L, _ = tf.unstack(tf.shape(y))

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = tf.reshape(q, (B, nhead, 1, head_dim))  # B, H, 1, 1 -> B, N, 1, h
        k = tf.reshape(k, (B, nhead, head_dim, L))  # B, H, L, 1 -> B, N, h, L
        v = tf.reshape(v, (B, nhead, head_dim, L))
        v = tf.transpose(v, perm=[0, 1, 3, 2])  # B, N, L, h

        pre_a = tf.matmul(q, k) / tf.sqrt(tf.cast(head_dim, q.dtype))
        if mask is not None:
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 1) # B, 1, 1, L_smask
            pre_a = tf.where(mask, -np.inf, pre_a)

        alphas = tf.nn.softmax(pre_a, axis=3)
        alphas = self.drop(alphas, training=training)  # B, N, 1, L
        att = tf.matmul(alphas, v)  # B, N, 1, h
        att = tf.reshape(att, (B, -1, 1, 1))  # B, N*h, 1, 1
        return self.WO(att)
