r"""Visualize t-SNE on trained car-ID model using Tensorboard

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import keras

from carid.models import resnet50
from carid.dataset import VeRiDataset


def parse_arguments():
  parser = argparse.ArgumentParser("Visualize t-SNE on VeRi dataset.")

  parser.add_argument(
    "--logdir", help="Log directory to store result",
    default=None, required=True)

  parser.add_argument(
    "--data_dir", help="path to dataset directory",
    default=None, required=True)

  parser.add_argument(
    "--num_points", help="Number of points per cluster.",
    type=int, default=50)

  return parser.parse_args()


def visualize():

  args = parse_arguments()

  # Load data
  veri_data = VeRiDataset(root_dir=args.data_dir).load()
  samples = veri_data.get_samples(num_samples=50, num_classes=10)

  # Define model
  model = resnet50().compile('sgd', 'mse')

  # Visualize
  callbacks = keras.callbacks.TensorBoard(
    log_dir=args.logdir,
    write_graph=False, embeddings_freq=1
  )


  return 0


if __name__ == '__main__':
  visualize()
