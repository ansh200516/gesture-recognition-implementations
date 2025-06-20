from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import tensorflow as tf
import numpy as np

from core.evaluate import accuracy


def _get_mixup_data(x, y, alpha=1.0, num_classes=1000):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))

    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    
    y_a = tf.one_hot(y, num_classes)
    y_b = tf.one_hot(tf.gather(y, index), num_classes)
    
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y


@tf.function
def train_step(config, x, y, model, criterion, optimizer):
    if config.AUG.MIXUP_PROB > 0.0:
        x, y = _get_mixup_data(x, y, config.AUG.MIXUP, config.MODEL.NUM_CLASSES)

    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        loss = criterion(y, outputs)

    gradients = tape.gradient(loss, model.trainable_variables)
    if config.TRAIN.CLIP_GRAD_NORM > 0.0:
        gradients, _ = tf.clip_by_global_norm(gradients, config.TRAIN.CLIP_GRAD_NORM)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, outputs


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to train mode')
    
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        loss, outputs = train_step(config, x, y, model, criterion, optimizer)

        losses.update(loss.numpy(), x.shape[0])

        if config.AUG.MIXUP_PROB > 0.0:
            y = tf.argmax(y, axis=1)

        prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.shape[0])
        top5.update(prec5, x.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = '=> Epoch[{0}][{1}/{2}]: ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=x.shape[0]/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logging.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        with writer.as_default():
            tf.summary.scalar('train_loss', losses.avg, step=global_steps)
            tf.summary.scalar('train_top1', top1.avg, step=global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


@tf.function
def test_step(x, y, model, criterion, valid_labels=None):
    outputs = model(x, training=False)
    if valid_labels:
        outputs = tf.gather(outputs, valid_labels, axis=1)
    
    loss = criterion(y, outputs)
    return loss, outputs


@tf.function
def distributed_test_step(x, y, model, criterion, replica_context, valid_labels=None):
    per_replica_loss, per_replica_outputs = replica_context.run(
        test_step, args=(x, y, model, criterion, valid_labels)
    )
    loss = replica_context.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
    return loss, per_replica_outputs


def test(config, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False, real_labels=None,
         valid_labels=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to eval mode')
    
    end = time.time()
    for i, (x, y) in enumerate(val_loader):
        
        if distributed:
            strategy = tf.distribute.get_strategy()
            loss, outputs = distributed_test_step(x, y, model, criterion, strategy, valid_labels)
        else:
            loss, outputs = test_step(x, y, model, criterion, valid_labels)

        if real_labels:
            # Re-implement RealLabels logic if needed for TF
            pass

        losses.update(loss.numpy(), x.shape[0])
        prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.shape[0])
        top5.update(prec5, x.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()
    
    # Distributed reduce is handled in distributed_test_step
    top1_acc, top5_acc, loss_avg = top1.avg, top5.avg, losses.avg
    
    if real_labels:
        # Re-implement RealLabels logic if needed for TF
        pass

    msg = '=> TEST:\t' \
          'Loss {loss_avg:.4f}\t' \
          'Error@1 {error1:.3f}%\t' \
          'Error@5 {error5:.3f}%\t' \
          'Accuracy@1 {top1:.3f}%\t' \
          'Accuracy@5 {top5:.3f}%\t'.format(
              loss_avg=loss_avg, top1=top1_acc,
              top5=top5_acc, error1=100-top1_acc,
              error5=100-top5_acc
          )
    logging.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        with writer.as_default():
            tf.summary.scalar('valid_loss', loss_avg, step=global_steps)
            tf.summary.scalar('valid_top1', top1_acc, step=global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    
    return top1_acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
