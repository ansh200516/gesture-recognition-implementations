from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    res = []
    for k in topk:
        # Ensure target is 1D tensor of integers
        if len(target.shape) > 1:
            target = tf.reshape(target, [-1])
        target = tf.cast(target, tf.int32)

        correct_k = tf.math.in_top_k(
            targets=target, predictions=output, k=k
        )
        accuracy_k = tf.reduce_mean(tf.cast(correct_k, tf.float32)) * 100.0
        res.append(accuracy_k.numpy())
    return res
