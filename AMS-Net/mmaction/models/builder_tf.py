import tensorflow as tf
from tensorflow.keras import layers

class Registry:
    """A registry to map strings to classes.
    Args:
        name (str): The name of the registry.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the class for a given key."""
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module_class (:obj:`type`): Module class to be registered.
        """
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered '
                           f'in {self.name}')
        self._module_dict[module_name] = module_class

    def register_module(self):
        """Decorator to register a module class."""
        def _register(cls):
            self._register_module(cls)
            return cls
        return _register

def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from a config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')

    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif issubclass(obj_type, layers.Layer):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid Keras Layer, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
            
    return obj_cls(**args)


BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
RECOGNIZERS = Registry('recognizer')
LOSSES = Registry('loss')
LOCALIZERS = Registry('localizer')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return tf.keras.Sequential(modules)
    return build_from_cfg(cfg, registry, default_args)

def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)

def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)

def build_recognizer(cfg):
    """Build recognizer."""
    return build(cfg, RECOGNIZERS)

def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)

def build_localizer(cfg):
    """Build localizer."""
    return build(cfg, LOCALIZERS)

def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS) 