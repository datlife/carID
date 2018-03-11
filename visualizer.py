r"""Visualize t-SNE on trained car-ID model using Tensorboard"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


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

  parser.add_argument(
    "--weights", help="path to pretrained weight file")

  return parser.parse_args()


if __name__ == '__main__':
  print('Hello')