import tensorflow as tf


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LabelSmoothingCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, epsilon=0.1, reduction=tf.keras.losses.Reduction.AUTO, name='label_smoothing_cross_entropy'):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(y_true, tf.shape(y_pred)[-1])

        n = tf.cast(tf.shape(y_pred)[-1], y_pred.dtype)
        log_preds = tf.nn.log_softmax(y_pred, axis=-1)
        loss = tf.reduce_mean(-tf.reduce_sum(log_preds, axis=-1))
        nll = -tf.reduce_sum(y_true * log_preds, axis=-1)
        nll = tf.reduce_mean(nll)
        return linear_combination(loss / n, nll, self.epsilon)


class SoftTargetCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='soft_target_cross_entropy'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.reduce_sum(-y_true * tf.nn.log_softmax(y_pred, axis=-1), axis=-1)
        return tf.reduce_mean(loss)


def build_criterion(config, train=True):
    if config.AUG.MIXUP_PROB > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = SoftTargetCrossEntropy() if train \
            else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif config.LOSS.LABEL_SMOOTHING > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = LabelSmoothingCrossEntropy(config.LOSS.LABEL_SMOOTHING)
    elif config.LOSS.LOSS == 'softmax':
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError('Unknown loss {}'.format(config.LOSS.LOSS))

    return criterion
