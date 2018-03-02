r"""Visualize t-SNE on trained car-ID model using Tensorboard

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import keras
import tensorflow as tf

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
  inputs   = tf.keras.layers.Input(shape=(224, 224, 3))
  resnet50 = tf.keras.applications.ResNet50(include_top=False)
  outputs  = tf.keras.layers.Lambda(lambda x: x, name="embedding")(resnet50(inputs))

  model = tf.keras.models.Model(inputs, outputs)
  model.compile('sgd', 'mse')
  model.load_weights(args.weights)
  model.summary()

  # Get embeddings
  samples = veri_data.get_samples(per_class_samples=50, num_classes=20)
  for s in samples:
    print(s, len(samples[s]), samples[s][0]['imageName'])
  return 0


def create_sprite_image(samples, output_path):
  """Create a giant image containing all the thumbnails for each sample.

  According to the TF documentation, sprite image needs to be in row-first order
  . For example,

        |1   |2  |3  |4  |
        |5   |6  |7  |8  |
        |......

  Args:
    samples:
    output_path:

  Returns:

  """
  pass


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
  visualize()
