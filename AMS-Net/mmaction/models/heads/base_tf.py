import tensorflow as tf
from abc import ABCMeta, abstractmethod

# A simple loss builder for tensorflow
def build_loss_tf(loss_cfg):
    loss_type = loss_cfg.get('type')
    if loss_type == 'CrossEntropyLoss':
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # Add other losses here as needed
    else:
        raise NotImplementedError(f'Loss type {loss_type} not implemented for tensorflow.')

class BaseHeadTF(tf.keras.layers.Layer, metaclass=ABCMeta):
    """Base class for head in TensorFlow.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``call``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Default: 0.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss_tf(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters."""

    @abstractmethod
    def call(self, x, **kwargs):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (tf.Tensor): The output of the model.
            labels (tf.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()

        if self.label_smooth_eps > 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(labels, cls_score, **kwargs)
        losses['loss_cls'] = loss_cls

        if not self.multi_class:
            top_1_acc = tf.keras.metrics.top_k_categorical_accuracy(labels, cls_score, k=1)
            top_5_acc = tf.keras.metrics.top_k_categorical_accuracy(labels, cls_score, k=5)
            losses['top1_acc'] = tf.reduce_mean(top_1_acc)
            losses['top5_acc'] = tf.reduce_mean(top_5_acc)

        return losses 