r"""Visualize t-SNE on trained car-ID model using Tensorboard

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import argparse
import numpy as np
import tensorflow as tf

from carid.dataset import VeRiDataset
from tensorboard.plugins import projector

NUM_CLASSES = 20
PER_CLASS = 50


def visualize():
  args = parse_arguments()

  train_dir = os.path.join(args.data_dir, VeRiDataset.TRAIN_DIR)
  projector_dir = os.path.join(args.logdir, 'projector')

  # Load data into memory
  veri_data = VeRiDataset(root_dir=args.data_dir).load()
  samples = veri_data.get_samples(
    per_class_samples=PER_CLASS,
    num_classes=NUM_CLASSES)

  # Load model
  inputs = tf.keras.layers.Input(
    shape=(224, 224, 3),
    name="input_img")

  resnet50 = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224, 224, 3))

  outputs = tf.keras.layers.Lambda(
    lambda x: x)(resnet50(inputs))

  model = tf.keras.models.Model(
    inputs,
    outputs)

  model.load_weights(args.weights)

  # Generate embeddings
  embeddings = []
  features, thumbnails = load_data(samples, train_dir, projector_dir)
  for i in range(len(features)):
    # shape = [1, 1, 2048]
    output = model.predict(np.expand_dims(features[i], 0))[0]
    embeddings.append(np.squeeze(np.squeeze(output, 0), 0))

  # Convert thumbnails to giant image
  sprite = images_to_sprite(np.array(thumbnails), rows=NUM_CLASSES)
  cv2.imwrite(os.path.join(projector_dir, 'sprite.png'), sprite)

  embedding_var = tf.Variable(
    tf.stack(embeddings, axis=0),
    trainable=False,
    name='embedding')
  with tf.Session() as sess:
    print(embedding_var)

    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(projector_dir)
    config = projector.ProjectorConfig()

    embed = config.embeddings.add()
    embed.tensor_name = embedding_var.name
    embed.metadata_path = os.path.join(projector_dir, 'metadata.tsv')
    embed.sprite.image_path = os.path.join(projector_dir, 'sprite.png')

    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver([embedding_var])
    saver.save(sess,
               os.path.join(args.logdir, 'model.ckpt'), global_step=1)


def load_data(samples, train_dir, projector_dir):
  meta_data = open(os.path.join(projector_dir, 'metadata.tsv'), 'w')
  features, thumbnails = [], []
  for group in samples:
    for instance in samples[group]:
      img_path = os.path.join(train_dir, instance['imageName'])
      image = cv2.resize(cv2.imread(img_path), (224, 224))
      meta_data.write('{}\n'.format(group))
      features.append(image)
      thumbnails.append(cv2.resize(image, (28, 28)))
  meta_data.close()
  return features, thumbnails


def images_to_sprite(thumbnails, rows):
  """Create a giant image containing all the thumbnails for each sample.
  """
  import numpy as np

  sprite = np.asarray(
    np.split(np.asarray(thumbnails), rows)).transpose((1, 0, 2, 3, 4))
  sprite = np.dstack(sprite)
  sprite = np.vstack(sprite)
  return sprite


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
