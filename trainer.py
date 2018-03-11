"""Train CarID model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from carid.models import carid_net
from carid.losses import triplet_loss
from carid.dataset import VeRiDataset

def train():
  # ##########################
  # Configure hyper-parameters
  # ##########################
  args = parse_args()
  batch_size = args.batch_size

  training_steps = args.steps
  steps_per_epoch = args.steps_per_epoch
  epochs_per_eval = 3
  training_epochs = int(training_steps // steps_per_epoch)

  cpu_cores = 8
  multi_gpu = True
  shuffle_buffer = 2e3

  # ########################
  # Load CIFAR-10 dataset
  # ########################
  veri_dataset = VeRiDataset(root_dir=args.dataset_dir)

  # ########################
  # Define a Classifier
  # ########################
  estimator = tf.estimator.Estimator(
      model_fn=carid_net(multi_gpu=multi_gpu),
      model_dir=args.model_dir,
      config=tf.estimator.RunConfig(),
      params={
          'learning_rate': 0.001,
          'optimizer': tf.train.AdamOptimizer,
          'multi_gpu': multi_gpu,
          'loss_function': triplet_loss,
          'margin': 0.2  # for loss calculation
      })

  # #########################
  # Training/Eval
  # #########################

  for _ in range(training_epochs // epochs_per_eval):
    train_data, eval_data = veri_dataset.split_training_data(
        test_size=0.3,
        shuffle=True)

    estimator.train(
        input_fn=lambda: veri_dataset.get_input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            data=train_data,
            epochs=None,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            num_parallel_calls=cpu_cores),
        steps=steps_per_epoch * epochs_per_eval,
        hooks=[])

    print("Start evaluating...")
    estimator.evaluate(
        input_fn=lambda: veri_dataset.get_input_fn(
            mode=tf.estimator.ModeKeys.EVAL,
            data=eval_data,
            epochs=1,
            batch_size=batch_size,
            shuffle_buffer=None,
            num_parallel_calls=cpu_cores))

  print("---- Training Completed ----")
  return 0


def parse_args():
  """Command line arguments.
  """
  parser = argparse.ArgumentParser(
      description="Train a Car Re-identification model")

  parser.add_argument(
      "--dataset_dir", help="Path to dataset directory.",
      default=None, required=True)

  parser.add_argument(
      "--batch_size", help="Number of training instances for every iteration",
      default=256, type=int)

  parser.add_argument(
      "--steps_per_epochs", help="Number of iteration per epochs",
      default=1e3, type=int)

  parser.add_argument(
      "--model_dir", help="Path to store training log and trained model",
      default=None)

  return parser.parse_args()


if __name__ == '__main__':
  train()
