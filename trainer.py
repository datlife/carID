"""Train CarID model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

from carid.losses import triplet_loss
from carid.dataset import VeRiDataset


def parse_args():
  """ Parse command line arguments.

  Returns
    args - arguments object
  """
  parser = argparse.ArgumentParser(
    description="Train a Car Re-identification model")
  parser.add_argument(
    "--training_labels", help="Path to training XML files.",
    default=None, required=True)
  parser.add_argument(
    "--batch_size", help="Number of training instances for every iteration",
    default=32)
  parser.add_argument(
    "--model_dir", help="Path to store log and trained model",
    default=None)
  return parser.parse_args()


def train():
  args = parse_args()

  # ##################
  # Define Dataset
  # ##################
  dataset = VeRiDataset(path=args.training_labels).load()

  # ##################
  # Define model
  # ##################
  inputs = [tf.keras.Input(
    shape=(None, None, 3),
    name="input_%s" % in_type)
    for in_type in ['anchor', 'positive', 'negative']]

  resnet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    input_shape=(None, None, 3))

  model = tf.keras.Model(
    inputs=inputs,
    outputs=[resnet50(input) for input in inputs],
    name="CarReId_net")

  model.compile(
    optimizer='adam',
    loss=triplet_loss(margin=0.2))

  estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model,
    model_dir=args.model_dir)

  # ##################
  # Train / Logging
  # ##################
  # @TODO: construct input_fn
  # @TODO: config augmentation during training/testing

  # ##################
  # Evaluate
  # ##################

  print("---- Training Completed ----")
  return 0


if __name__ == '__main__':
  train()
