import os
import os.path as osp
import pickle

import tensorflow as tf

# In a real scenario, initialize and use a distributed training framework
# like horovod or tf.distribute. This is a placeholder for single-threaded execution.
class Dist:
    """A mock class for distributed training information."""
    
    @staticmethod
    def get_dist_info():
        """Returns rank and world size."""
        if 'TF_CONFIG' in os.environ:
            # For tf.distribute.MultiWorkerMirroredStrategy
            import json
            tf_config = json.loads(os.environ['TF_CONFIG'])
            task = tf_config.get('task', {})
            cluster = tf_config.get('cluster', {})
            worker = cluster.get('worker', [])
            if task and worker:
                rank = task.get('index')
                world_size = len(worker)
                return rank, world_size
        try:
            # For Horovod
            import horovod.tensorflow as hvd
            if not hvd.is_initialized():
                hvd.init()
            return hvd.rank(), hvd.size()
        except ImportError:
            return 0, 1

    @staticmethod
    def barrier():
        """A barrier for synchronization."""
        try:
            import horovod.tensorflow as hvd
            if not hvd.is_initialized():
                hvd.init()
            hvd.allreduce(tf.constant(0), name="barrier")
        except ImportError:
            pass


def mkdir_or_exist(dir_name, mode=0o777):
    """Create a directory if it doesn't exist."""
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class LFBInferHead(tf.keras.layers.Layer):
    """Long-Term Feature Bank Infer Head in TensorFlow.

    This head is used to derive and save the LFB without affecting the input.
    The data format is assumed to be `(N, T, H, W, C)`.

    Args:
        lfb_prefix_path (str): The prefix path to store the lfb.
        dataset_mode (str, optional): Which dataset to be inferred. Choices are
            'train', 'val' or 'test'. Default: 'train'.
        use_half_precision (bool, optional): Whether to store the
            half-precision roi features. Default: True.
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
    """

    def __init__(self,
                 lfb_prefix_path,
                 dataset_mode='train',
                 use_half_precision=True,
                 temporal_pool_type='avg',
                 spatial_pool_type='max',
                 **kwargs):
        super().__init__(**kwargs)

        rank, _ = Dist.get_dist_info()
        if rank == 0:
            if not osp.exists(lfb_prefix_path):
                print(f'lfb prefix path {lfb_prefix_path} does not exist. '
                      'Creating the folder...')
                mkdir_or_exist(lfb_prefix_path)
            print('\nInferring LFB...')

        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.lfb_prefix_path = lfb_prefix_path
        self.dataset_mode = dataset_mode
        self.use_half_precision = use_half_precision
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.all_features = []
        self.all_metadata = []

    def call(self, x, rois, img_metas):
        """
        Args:
            x (tf.Tensor): The input features, shape (N, T, H, W, C).
            rois (tf.Tensor): The rois.
            img_metas (list[dict]): A list of image meta information.
        """
        # Temporal pooling over T dim (axis 1)
        if self.temporal_pool_type == 'avg':
            features = tf.reduce_mean(x, axis=1, keepdims=True)
        else:
            features = tf.reduce_max(x, axis=1, keepdims=True)

        # Spatial pooling over H, W dims (axis 2, 3)
        if self.spatial_pool_type == 'avg':
            features = tf.reduce_mean(features, axis=[2, 3], keepdims=True)
        else:
            features = tf.reduce_max(features, axis=[2, 3], keepdims=True)

        if self.use_half_precision:
            features = tf.cast(features, dtype=tf.float16)

        inds = tf.cast(rois[:, 0], dtype=tf.int32)
        for ind in inds.numpy():
            self.all_metadata.append(img_metas[ind]['img_key'])

        self.all_features.extend(list(features.numpy()))

        return x

    def finalize_and_save(self):
        """
        This method should be called after inference on all data is complete
        to save the features.
        """
        assert len(self.all_features) == len(self.all_metadata), (
            'features and metadata are not equal in length!')

        rank, world_size = Dist.get_dist_info()
        if world_size > 1:
            Dist.barrier()

        _lfb = {}
        for feature, metadata in zip(self.all_features, self.all_metadata):
            video_id, timestamp = metadata.split(',')
            timestamp = int(timestamp)

            if video_id not in _lfb:
                _lfb[video_id] = {}
            if timestamp not in _lfb[video_id]:
                _lfb[video_id][timestamp] = []

            _lfb[video_id][timestamp].append(feature.squeeze())

        _lfb_file_path = osp.normpath(
            osp.join(self.lfb_prefix_path,
                     f'_lfb_{self.dataset_mode}_{rank}.pkl'))

        with open(_lfb_file_path, 'wb') as f:
            pickle.dump(_lfb, f)

        print(f'{len(self.all_features)} features from {len(_lfb)} videos '
              f'on GPU {rank} have been stored in {_lfb_file_path}.')

        if world_size > 1:
            Dist.barrier()
        if rank > 0:
            return

        print('Gathering all the roi features...')

        lfb = {}
        for rank_id in range(world_size):
            _lfb_file_path = osp.normpath(
                osp.join(self.lfb_prefix_path,
                         f'_lfb_{self.dataset_mode}_{rank_id}.pkl'))

            with open(_lfb_file_path, 'rb') as f:
                _lfb = pickle.load(f)

            for video_id in _lfb:
                if video_id not in lfb:
                    lfb[video_id] = _lfb[video_id]
                else:
                    lfb[video_id].update(_lfb[video_id])

        lfb_file_path = osp.normpath(
            osp.join(self.lfb_prefix_path, f'lfb_{self.dataset_mode}.pkl'))
        with open(lfb_file_path, 'wb') as f:
            pickle.dump(lfb, f)

        print(f'LFB has been constructed in {lfb_file_path}!') 