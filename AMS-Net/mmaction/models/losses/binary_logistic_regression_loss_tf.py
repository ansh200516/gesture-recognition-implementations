import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..registry import LOSSES


def binary_logistic_regression_loss(reg_score,
                                    label,
                                    threshold=0.5,
                                    ratio_range=(1.05, 21),
                                    eps=1e-5):
    """Binary Logistic Regression Loss."""
    label = tf.reshape(label, [-1])
    reg_score = tf.reshape(reg_score, [-1])

    pmask = tf.cast(label > threshold, dtype=tf.float32)
    num_positive = tf.maximum(tf.reduce_sum(pmask), 1.0)
    num_entries = tf.cast(tf.shape(label)[0], dtype=tf.float32)
    ratio = num_entries / num_positive
    # clip ratio value between ratio_range
    ratio = tf.clip_by_value(ratio, ratio_range[0], ratio_range[1])

    coef_0 = 0.5 * ratio / (ratio - 1.0)
    coef_1 = 0.5 * ratio
    loss = coef_1 * pmask * tf.math.log(reg_score + eps) + coef_0 * (
        1.0 - pmask) * tf.math.log(1.0 - reg_score + eps)
    loss = -tf.reduce_mean(loss)
    return loss


@LOSSES.register_module()
class BinaryLogisticRegressionLoss(Layer):
    """Binary Logistic Regression Loss.

    It will calculate binary logistic regression loss given reg_score and
    label.
    """

    def call(self,
             reg_score,
             label,
             threshold=0.5,
             ratio_range=(1.05, 21),
             eps=1e-5):
        """Calculate Binary Logistic Regression Loss.

        Args:
                reg_score (tf.Tensor): Predicted score by model.
                label (tf.Tensor): Groundtruth labels.
                threshold (float): Threshold for positive instances.
                    Default: 0.5.
                ratio_range (tuple): Lower bound and upper bound for ratio.
                    Default: (1.05, 21)
                eps (float): Epsilon for small value. Default: 1e-5.

        Returns:
                tf.Tensor: Returned binary logistic loss.
        """

        return binary_logistic_regression_loss(reg_score, label, threshold,
                                               ratio_range, eps)