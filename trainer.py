"""Train CarID model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf

from carid.losses import triplet_loss
from carid.dataset import VeRiDataset

tf.logging.set_verbosity(tf.logging.DEBUG)
# @TODO: config augmentation during training/testing


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

def train():
  args = parse_args()

  veri_dataset = VeRiDataset(root_dir=args.dataset_dir).load()

  training_input_fn, validataion_input_fn = veri_dataset.get_input_fn(
      is_training=True,
      batch_size=args.batch_size,
      shuffle=True)

  model = build_model()
  model.compile(optimizer='adam', loss=triplet_loss(margin=0.2))

  # Train / Logging
  callbacks = [
    tf.keras.callbacks.TensorBoard(
      log_dir=args.model_dir,
      write_graph=False),

    tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(args.model_dir,'{epoch:02d}-{val_loss:.2f}.hdf5'),
      monitor='val_loss',
      save_best_only=True
    )]

  model.fit_generator(
    epochs=args.steps,
    steps_per_epoch=100,
    generator=keras_generator(input_fn=training_input_fn),
    validation_data=keras_generator(input_fn=validataion_input_fn),
    validation_steps=100,
    callbacks=callbacks,
    workers=0,
    verbose=1)

  # ##################
  # Evaluate
  # ##################

  print("---- Training Completed ----")
  return 0

def keras_generator(input_fn):
  K = tf.keras.backend

  while True:
    yield K.get_session().run(input_fn)

def build_model():
  inputs = [tf.keras.Input(
    shape=(224, 224, 3),
    name="input_%s" % in_type)
    for in_type in ['anchor', 'positive', 'negative']]

  resnet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    input_shape=(224, 224, 3))

  outputs = tf.keras.layers.concatenate(
    [resnet50(input) for input in inputs],
    name='output')

  model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="CarReId_net")
  return model

if __name__ == '__main__':
  train()


  # estimator = tf.keras.estimator.model_to_estimator(
  #   keras_model=model,
  #   model_dir=args.model_dir)
  #
  # estimator.train(
  #   steps=args.steps,
  #   input_fn=lambda: veri_dataset.get_input_fn(
  #     is_training=True,
  #     batch_size=args.batch_size,
  #     shuffle=True),
  # )
