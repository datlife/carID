"""Train CarID model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf

from carid.models import resnet_carid
from carid.losses import triplet_loss
from carid.dataset import VeRiDataset
from carid.ProgressBar import ProgressBarHook
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def main():
  # ##########################
  # Configure hyper-parameters
  # ##########################
  args = parse_args()
  batch_size = 16
  training_steps = args.steps
  steps_per_epoch = args.steps_per_epoch
  epochs_per_eval = 3
  training_epochs = int(training_steps // steps_per_epoch)

  cpu_cores = 8
  multi_gpu = False
  shuffle_buffer = 128

  # ########################
  # Load VeRi dataset
  # ########################
  veri_dataset = VeRiDataset(root_dir=args.dataset_dir).load()

  # ########################
  # Define a Classifier
  # ########################
  estimator = tf.estimator.Estimator(
      model_fn=resnet_carid(multi_gpu=multi_gpu),
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
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            num_parallel_calls=cpu_cores),
        steps=steps_per_epoch * epochs_per_eval,
        hooks=[ProgressBarHook(epochs=10, steps_per_epoch=steps_per_epoch)])

  #   print("Start evaluating...")
  #   estimator.evaluate(
  #       input_fn=lambda: veri_dataset.get_input_fn(
  #           mode=tf.estimator.ModeKeys.EVAL,
  #           data=eval_data,
  #           batch_size=batch_size,
  #           shuffle_buffer=None,
  #           num_parallel_calls=cpu_cores))

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
    "--steps", help="Number of iteration per epochs",
    default=30e3, type=int)

  parser.add_argument(
      "--steps_per_epoch", help="Number of iteration per epochs",
      default=1e3, type=int)

  parser.add_argument(
      "--model_dir", help="Path to store training log and trained model",
      default=None)

  return parser.parse_args()


if __name__ == '__main__':
  main()
