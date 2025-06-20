from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers


def _is_depthwise(layer):
    return isinstance(layer, layers.DepthwiseConv2D)


def set_wd(cfg, model):
    """
    Identifies variables that should be excluded from weight decay.

    In TensorFlow, rather than creating parameter groups, we identify variables
    to exclude and pass them to an optimizer that supports such functionality,
    like the AdamW optimizer from TensorFlow Model Garden.

    Args:
        cfg: The configuration object.
        model: The Keras model.

    Returns:
        A list of variable names to be excluded from weight decay.
    """
    without_decay_list = cfg.TRAIN.WITHOUT_WD_LIST
    vars_to_exclude = []

    # The original PyTorch implementation supported custom attributes on the model
    # to specify parameters to exclude from weight decay. We replicate this
    # by checking for specific attributes on the Keras model.
    skip_patterns = []
    if hasattr(model, 'no_weight_decay'):
        skip_patterns = model.no_weight_decay()

    skip_keywords = []
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    for var in model.trainable_variables:
        var_name = var.name
        if any(pattern in var_name for pattern in skip_patterns) or \
           any(keyword in var_name for keyword in skip_keywords):
            if cfg.VERBOSE:
                print(f'=> set {var_name} wd to 0')
            vars_to_exclude.append(var_name)

    for layer in model.layers:
        layer_vars = [v.name for v in layer.trainable_variables]
        
        if _is_depthwise(layer) and 'dw' in without_decay_list:
            if cfg.VERBOSE:
                print(f'=> Excluding depthwise conv weights of {layer.name} from weight decay.')
            vars_to_exclude.extend(layer_vars)
        elif isinstance(layer, layers.BatchNormalization) and 'bn' in without_decay_list:
            if cfg.VERBOSE:
                print(f'=> Excluding bn weights of {layer.name} from weight decay.')
            vars_to_exclude.extend(layer_vars)
        elif isinstance(layer, layers.GroupNormalization) and 'gn' in without_decay_list:
            if cfg.VERBOSE:
                print(f'=> Excluding gn weights of {layer.name} from weight decay.')
            vars_to_exclude.extend(layer_vars)
        elif isinstance(layer, layers.LayerNormalization) and 'ln' in without_decay_list:
            if cfg.VERBOSE:
                print(f'=> Excluding ln weights of {layer.name} from weight decay.')
            vars_to_exclude.extend(layer_vars)

    if 'bias' in without_decay_list:
        for var in model.trainable_variables:
            if 'bias' in var.name:
                if var.name not in vars_to_exclude:
                    if cfg.VERBOSE:
                        print(f'=> Excluding bias({var.name}) from weight decay.')
                    vars_to_exclude.append(var.name)

    return list(set(vars_to_exclude))


def build_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'timm':
        raise ValueError("Timm optimizer is not supported in TensorFlow.")

    optimizer = None
    lr = cfg.TRAIN.LR
    wd = cfg.TRAIN.WD

    if cfg.TRAIN.OPTIMIZER == 'sgd':
        print("WARNING: tf.keras.optimizers.SGD does not support weight decay directly. "
              "Consider using AdamW or adding L2 regularization at the layer level.")
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=cfg.TRAIN.MOMENTUM,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        print("WARNING: tf.keras.optimizers.Adam does not support weight decay. "
              "Consider using AdamW.")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamW':
        try:
            from tf_models.optimization import AdamW as AdamW_TFModels
        except ImportError:
            print("WARNING: tf-models-official not installed. Using tf.keras.optimizers.AdamW "
                  "without weight decay exclusion. Install with: pip install tf-models-official")
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
            
        vars_to_exclude = set_wd(cfg, model)
        if cfg.VERBOSE:
            print(f"Excluding {len(vars_to_exclude)} variables from weight decay.")

        optimizer = AdamW_TFModels(
            weight_decay_rate=wd,
            learning_rate=lr,
            exclude_from_weight_decay=vars_to_exclude,
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        print("WARNING: tf.keras.optimizers.RMSprop does not support weight decay directly. "
              "Consider using AdamW or adding L2 regularization at the layer level.")
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=cfg.TRAIN.RMSPROP_ALPHA,
            momentum=cfg.TRAIN.MOMENTUM,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer

