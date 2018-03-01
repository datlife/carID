r"""Visualize t-SNE on trained car-ID model using Tensorboard

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import keras
import tensorflow as tf
from tensorboard.plugins.projector import projector_plugin

from carid.models import resnet50
from carid.dataset import VeRiDataset


def construct_projector_summary(args):
  config = projector_plugin.ProjectorConfig()
  embed = config.embeddings.add()
  embed.tensor_name = 'embedding:0'
  embed.metadata_path = os.path.join(
    args.logdir, 'projector/metadata.tsv')
  embed.sprite.image_path = os.path.join(
    args.model_dir, 'veri_sprite_10k.png')
  embed.sprite.single_image_dim.extend([28, 28])


def visualize():
  args = parse_arguments()

  # Load data
  veri_data = VeRiDataset(root_dir=args.data_dir).load()

  # Load model
  inputs = tf.keras.layers.Input(shape=(224, 224, 3))
  model = keras.models.Model(
    inputs=inputs,
    outputs=tf.keras.layers.Lambda(
      lambda x: x, "embedding")(resnet50()(inputs)))
  model.compile('sgd', 'mse')
  model.summary()

  # Get embeddings
  samples = veri_data.get_samples(num_samples=50, num_classes=10)


  return 0


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


if __name__ == '__main__':
  visualize()
