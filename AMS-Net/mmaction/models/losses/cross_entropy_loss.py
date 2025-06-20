import tensorflow as tf
from ..registry import LOSSES
from .base_tf import BaseWeightedLoss


@LOSSES.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None, **kwargs):
        super().__init__(loss_weight=loss_weight, **kwargs)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = tf.constant(class_weight, dtype=tf.float32)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (tf.Tensor): The class score.
            label (tf.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            tf.Tensor: The returned CrossEntropy loss.
        """
        if len(cls_score.shape) == len(label.shape) and cls_score.shape == label.shape:
            # calculate loss for soft label
            assert len(cls_score.shape) == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = tf.nn.log_softmax(cls_score, axis=1)
            if self.class_weight is not None:
                lsm = lsm * tf.expand_dims(self.class_weight, 0)
            loss_cls = -tf.reduce_sum(label * lsm, axis=1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                loss_cls = tf.reduce_sum(loss_cls) / tf.reduce_sum(
                    tf.expand_dims(self.class_weight, 0) * label)
            else:
                loss_cls = tf.reduce_mean(loss_cls)
        else:
            # calculate loss for hard label
            loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(label, dtype=tf.int32), logits=cls_score)

            if self.class_weight is not None:
                weight = tf.gather(self.class_weight, tf.cast(label, dtype=tf.int32))
                loss_cls = loss_cls * weight
                loss_cls = tf.reduce_mean(loss_cls) 
            else:
                loss_cls = tf.reduce_mean(loss_cls)

        return loss_cls


@LOSSES.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None, **kwargs):
        super().__init__(loss_weight=loss_weight, **kwargs)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = tf.constant(class_weight, dtype=tf.float32)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (tf.Tensor): The class score.
            label (tf.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            tf.Tensor: The returned bce loss with logits.
        """
        loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, dtype=tf.float32), logits=cls_score)
        if self.class_weight is not None:
             loss_cls = loss_cls * self.class_weight
        
        return tf.reduce_mean(loss_cls)
