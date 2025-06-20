import tensorflow as tf
from .ams_resnet3d_slowfast_tf import AMSResNet3dPathway

class AMSResNet3dSlowOnlyTF(AMSResNet3dPathway):
    """SlowOnly backbone based on AMSResNet3dPathway in TensorFlow.

    Args:
        *args (arguments): Arguments same as :class:`AMSResNet3dPathway`.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: (1, 7, 7).
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Keywords arguments for
            :class:`AMSResNet3dPathway`.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 conv1_kernel=(1, 7, 7),
                 conv1_stride_t=1,
                 pool1_stride_t=1,
                 inflate=(0, 0, 1, 1),
                 with_pool2=False,
                 **kwargs):
        super().__init__(
            *args,
            lateral=lateral,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            **kwargs)

        assert not self.lateral 