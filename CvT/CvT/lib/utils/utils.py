from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import timedelta
from pathlib import Path

import os
import logging
import shutil
import time

import tensorflow as tf
import numpy as np

from utils.comm import comm


def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.txt'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:[P:%(process)d]:' + comm.head + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    dataset = cfg.DATASET.DATASET
    cfg_name = cfg.NAME

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {} ...'.format(root_output_dir))
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print('=> setup logger ...')
    setup_logger(final_output_dir, comm.rank, phase)

    return str(final_output_dir)


def init_distributed(args):
    strategy = None
    if 'TF_CONFIG' in os.environ:
        args.distributed = True
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        
        import json
        tf_config = json.loads(os.environ['TF_CONFIG'])
        comm.rank = tf_config['task']['index']
        comm.world_size = len(tf_config['cluster']['worker'])
        if hasattr(args, 'local_rank'):
            comm.local_rank = args.local_rank
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[args.local_rank], 'GPU')
        
        logging.info("=> Multi-worker distributed training initialized.")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        args.num_gpus = len(gpus)
        args.distributed = args.num_gpus > 1
        if args.distributed:
            logging.info(f"=> Multi-GPU single machine training on {args.num_gpus} GPUs.")
            strategy = tf.distribute.MirroredStrategy()
        else:
            logging.info("=> Single device training (GPU or CPU).")
            strategy = tf.distribute.get_strategy()

    comm.set_strategy(strategy)
    return strategy


def setup_cudnn(config):
    if config.CUDNN.DETERMINISTIC:
        tf.config.experimental.enable_op_determinism()

    if config.CUDNN.BENCHMARK:
        logging.info("TensorFlow does not have a direct equivalent to 'cudnn.benchmark = True'. Performance tuning is often automatic.")

    if config.CUDNN.ENABLED:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            logging.warning("CuDNN is enabled in config, but no GPUs were found.")
        else:
            logging.info(f"CuDNN enabled, found {len(gpus)} GPUs.")


def count_parameters(model):
    params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    return params/1000000


def summary_model_on_master(model, config, output_dir, copy):
    if comm.is_main_process():
        # Keras model.summary() provides a good overview.
        # It needs to be captured from stdout.
        from io import StringIO
        import sys

        summary_string = StringIO()
        original_stdout = sys.stdout
        sys.stdout = summary_string
        model.summary(
            line_length=120, 
            print_fn=lambda x: summary_string.write(x + '\n')
        )
        sys.stdout = original_stdout
        
        logging.info(f'\n{summary_string.getvalue()}')

        try:
            num_params = count_parameters(model)
            logging.info("Trainable Model Total Parameter: \t%2.1fM" % num_params)
        except Exception as e:
            logging.error(f'=> error when counting parameters: {e}')


def resume_checkpoint(model,
                      optimizer,
                      config,
                      output_dir,
                      in_epoch):
    best_perf = 0.0
    begin_epoch_or_step = 0

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=output_dir, max_to_keep=3
    )
    
    resume_path = config.TRAIN.CHECKPOINT if config.TRAIN.CHECKPOINT else manager.latest_checkpoint

    if config.TRAIN.AUTO_RESUME and resume_path:
        logging.info(
            "=> loading checkpoint '{}'".format(resume_path)
        )
        status = checkpoint.restore(resume_path)
        status.assert_consumed()  # Optional: check that all variables were restored.

        # To get metadata like epoch and perf, we would need to save it separately
        # or parse it from the checkpoint path if we include it in the filename.
        # For simplicity, we assume epoch can be parsed from path.
        try:
            if in_epoch:
                begin_epoch_or_step = int(resume_path.split('-')[-1])
            else: # step based not clearly supported here, assuming epoch
                 begin_epoch_or_step = int(resume_path.split('-')[-1]) * config.STEPS_PER_EPOCH
        except:
             logging.warning("Could not parse epoch from checkpoint path. Starting from 0.")
             begin_epoch_or_step = 0
        
        # 'best_perf' would also need to be stored, e.g. in a separate JSON file.
        # For now, we don't restore it.
        logging.info(
            "=> {}: loaded checkpoint '{}' ({}: {})"
            .format(comm.head,
                    resume_path,
                    'epoch' if in_epoch else 'step',
                    begin_epoch_or_step)
        )

    return best_perf, begin_epoch_or_step


def save_checkpoint_on_master(model,
                              *,
                              distributed, # distributed is not used in TF2 style
                              model_name,
                              optimizer,
                              output_dir,
                              in_epoch,
                              epoch_or_step,
                              best_perf):
    if not comm.is_main_process():
        return

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=output_dir, max_to_keep=3
    )

    logging.info('=> saving checkpoint to {}'.format(output_dir))
    
    try:
        save_path = manager.save(checkpoint_number=epoch_or_step)
        logging.info(f"=> saved checkpoint for epoch {epoch_or_step}: {save_path}")
        # Here we could also save best_perf to a file.
    except Exception as e:
        logging.error(f'=> error when saving checkpoint! {e}')


def save_model_on_master(model, distributed, out_dir, fname):
    if not comm.is_main_process():
        return

    try:
        fname_full = os.path.join(out_dir, fname)
        logging.info(f'=> save model weights to {fname_full}')
        model.save_weights(fname_full)
    except Exception as e:
        logging.error(f'=> error when saving model weights! {e}')


def strip_prefix_if_present(model_weights, prefix):
    """
    In TensorFlow, loading weights with different prefixes is handled by
    loading weights `by_name`. If a part of a model is saved and needs to be
    loaded into a larger model, one can iterate through layers and load
    weights manually if names match. A direct 'strip_prefix' is less common.
    This function can be adapted if a specific name mapping is needed.
    """
    logging.warning("'strip_prefix_if_present' is a PyTorch-specific utility and has no direct TF equivalent. Weight loading in TF is typically done 'by_name'.")
    return model_weights
