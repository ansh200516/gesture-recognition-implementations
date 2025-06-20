from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False


def build_lr_scheduler(cfg, steps_per_epoch):
    """
    Builds a learning rate scheduler for TensorFlow/Keras.

    Args:
        cfg: A config object.
        steps_per_epoch: The number of training steps per epoch.

    Returns:
        A tf.keras.optimizers.schedules.LearningRateSchedule instance.
        Returns None if no scheduler is specified.
    """
    if 'METHOD' not in cfg.TRAIN.LR_SCHEDULER:
        # It's possible to not use a scheduler.
        return None

    method = cfg.TRAIN.LR_SCHEDULER.METHOD
    if method is None or method.lower() == 'none':
        return None

    # This base learning rate is expected to be configured.
    # It's used as the initial value for most schedulers.
    initial_lr = cfg.TRAIN.LR

    if method == 'MultiStep':
        boundaries = [
            epoch * steps_per_epoch for epoch in cfg.TRAIN.LR_SCHEDULER.MILESTONES
        ]
        values = [
            initial_lr * (cfg.TRAIN.LR_SCHEDULER.GAMMA ** i)
            for i in range(len(boundaries) + 1)
        ]
        lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=values
        )
    elif method == 'CosineAnnealing':
        decay_steps = cfg.TRAIN.END_EPOCH * steps_per_epoch
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=cfg.TRAIN.LR_SCHEDULER.ETA_MIN / initial_lr
        )
    elif method == 'CyclicLR':
        if not TFA_AVAILABLE:
            raise ImportError(
                "CyclicLR requires 'tensorflow-addons'. "
                "Please install it with 'pip install tensorflow-addons'"
            )
        lr_scheduler = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=cfg.TRAIN.LR_SCHEDULER.BASE_LR,
            maximal_learning_rate=cfg.TRAIN.LR_SCHEDULER.MAX_LR,
            step_size=cfg.TRAIN.LR_SCHEDULER.STEP_SIZE_UP,
            scale_mode='cycle',
            name='CyclicLearningRate'
        )
    elif method == 'timm':
        raise ValueError(
            "'timm' scheduler is PyTorch-specific and not supported in this "
            "TensorFlow version. Please choose a different scheduler method "
            "or implement a TensorFlow-equivalent."
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(method))

    return lr_scheduler

