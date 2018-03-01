"""Train CarID model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf

from carid.models import resnet50
from carid.losses import triplet_loss
from carid.dataset import VeRiDataset
from carid.metrics import ap_distance, an_distance

tf.logging.set_verbosity(tf.logging.DEBUG)
# @TODO: config augmentation during training/testing


def train():
  args = parse_args()
  veri_dataset = VeRiDataset(root_dir=args.dataset_dir).load()

  # Load model
  model = resnet50()
  model.compile(
    optimizer='adam',
    loss=triplet_loss(margin=0.2),
    metrics=[ap_distance, an_distance])
  model.summary()

  # Train / Logging
  train_input_fn, val_input_fn = veri_dataset.get_input_fn(
      is_training=True,
      batch_size=args.batch_size,
      shuffle=True,
      buffer_size=100)

  model.fit_generator(
    generator=keras_generator(input_fn=train_input_fn),
    validation_data=keras_generator(input_fn=val_input_fn),
    epochs=args.steps,
    steps_per_epoch=50,
    validation_steps=50,
    callbacks=keras_callbacks(logdir=args.model_dir),
    workers=0,
    verbose=1)

  print("---- Training Completed ----")
  return 0


def keras_generator(input_fn):
  K = tf.keras.backend
  while True:
    yield K.get_session().run(input_fn)


def keras_callbacks(logdir):
  callbacks = [
    tf.keras.callbacks.TensorBoard(
      log_dir=logdir,
      write_graph=False),
    tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(logdir, 'carid.weights'),
      monitor='val_loss',
      save_best_only=True,
      save_weights_only=True)]

  return callbacks


def parse_args():
  """Command line arguments.
  """
  parser = argparse.ArgumentParser(
    description="Train a Car Re-identification model")

  parser.add_argument(
    "--dataset_dir", help="Path to dataset directory.",
    default=None, required=True)

  parser.add_argument(
    "--steps", help="Number of training steps",
    default=1, type=int)
  parser.add_argument(
    "--batch_size", help="Number of training instances for every iteration",
    default=32, type=int)

  parser.add_argument(
    "--model_dir", help="Path to store log and trained model",
    default=None)
  return parser.parse_args()


if __name__ == '__main__':
  train()
