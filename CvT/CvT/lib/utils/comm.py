import os
import json
import pickle
import tensorflow as tf

class Comm:
    """
    A communication helper class for distributed training in TensorFlow.
    It relies on the TF_CONFIG environment variable for multi-worker setup
    and provides utility methods to access rank, world size, etc.
    """
    def __init__(self):
        self._strategy = None
        self.rank = 0
        self.world_size = 1
        self._local_rank = 0

        if 'TF_CONFIG' in os.environ:
            try:
                tf_config = json.loads(os.environ['TF_CONFIG'])
                if 'task' in tf_config and 'type' in tf_config['task'] and 'index' in tf_config['task']:
                    # This is a worker process
                    self.rank = tf_config['task']['index']
                    self.world_size = len(tf_config['cluster'].get('worker', []))
            except (ValueError, KeyError):
                print("Warning: Could not parse TF_CONFIG.")
                pass
    
    def set_strategy(self, strategy):
        self._strategy = strategy

    @property
    def local_rank(self):
        return self._local_rank
    
    @local_rank.setter
    def local_rank(self, value):
        self._local_rank = value

    @property
    def head(self):
        return f'Rank[{self.rank}/{self.world_size}]'

    def is_main_process(self):
        return self.rank == 0

    def synchronize(self):
        """
        A barrier to synchronize all workers in distributed training.
        """
        if self.world_size > 1 and self._strategy and hasattr(self._strategy.extended, 'barrier'):
            self._strategy.extended.barrier()
        elif self.world_size > 1 and self._strategy:
             # Fallback for strategies without an explicit barrier.
             # An all-reduce on a dummy tensor can act as a barrier.
            dummy_tensor = tf.constant(1.0)
            self._strategy.reduce(tf.distribute.ReduceOp.SUM, dummy_tensor, axis=None)

comm = Comm()


def all_gather(data, strategy=None):
    """
    Run all_gather on arbitrary picklable data.
    This is a collective operation that needs to be run inside a `strategy.run` call.
    """
    if comm.world_size == 1:
        return [data]

    current_strategy = strategy if strategy else comm._strategy
    if not current_strategy or not isinstance(current_strategy, tf.distribute.MultiWorkerMirroredStrategy):
        if comm.is_main_process():
            print("Warning: all_gather is implemented for MultiWorkerMirroredStrategy but the current strategy is different or not set.")
        # Fallback for non-multiworker or no strategy.
        return [data] * comm.world_size

    def inner_all_gather():
        # 1. Pickle and convert to tensor
        buffer = pickle.dumps(data)
        local_tensor = tf.constant(list(buffer), dtype=tf.uint8)
        local_size = tf.shape(local_tensor)[0]

        # 2. Gather sizes of all tensors
        all_sizes = current_strategy.gather(local_size, axis=0)
        max_size = tf.reduce_max(all_sizes)

        # 3. Pad local tensor to max size
        pad_size = max_size - local_size
        padding = tf.zeros([pad_size], dtype=tf.uint8)
        padded_local_tensor = tf.concat([local_tensor, padding], axis=0)

        # 4. All-gather the padded tensors
        all_padded_tensors = current_strategy.gather(padded_local_tensor, axis=0)

        return all_padded_tensors, all_sizes

    # This function needs to be run in the context of each replica.
    all_padded_tensors_replica, all_sizes_replica = current_strategy.run(inner_all_gather)

    # The results are per-replica. We need to process them on the host.
    if comm.is_main_process():
        data_list = []
        # `current_strategy.experimental_local_results` gives access to values from each local replica
        tensors_list = current_strategy.experimental_local_results(all_padded_tensors_replica)
        sizes_list = current_strategy.experimental_local_results(all_sizes_replica)

        # Assuming one replica per worker for simplicity here.
        # This part might need adjustment for multiple replicas per worker.
        for tensors, sizes in zip(tensors_list, sizes_list):
            for i in range(tf.shape(tensors)[0]):
                size = sizes[i]
                buffer = tensors[i, :size].numpy().tobytes()
                data_list.append(pickle.loads(buffer))
        return data_list
    else:
        # Other processes participate but don't need to compute the final list
        return []


def reduce_dict(input_dict, average=True, strategy=None):
    """
    Reduce the values in the dictionary from all processes so that the main process
    has the averaged results.
    """
    if comm.world_size < 2:
        return input_dict

    current_strategy = strategy if strategy else comm._strategy
    if not current_strategy:
        return input_dict

    names = []
    values = []
    for k in sorted(input_dict.keys()):
        names.append(k)
        # Ensure values are tensors
        values.append(tf.convert_to_tensor(input_dict[k], dtype=tf.float32))

    stacked_values = tf.stack(values, axis=0)

    def step_fn(x):
        return current_strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None)

    reduced_values = current_strategy.run(step_fn, args=(stacked_values,))
    
    # On the main process, get the result and process it
    if comm.is_main_process():
        # With strategies, the result of reduce is a PerReplica object.
        # We can get the value from the first replica.
        reduced_tensor = current_strategy.experimental_local_results(reduced_values)[0]

        if average:
            reduced_tensor /= comm.world_size
        
        return {k: v for k, v in zip(names, reduced_tensor)}
    else:
        # On other workers, return a dict of zeros with the same structure
        return {k: tf.zeros_like(v) for k, v in zip(names, values)}

