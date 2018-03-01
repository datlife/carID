r""" Triplet Loss Function

This file contains the definition of Triplet Loss

"""
import tensorflow as tf


def triplet_loss(margin=0.2):
  """Triplet Loss Implementation

  TripletLoss = Max(0.0, d_ap - d_an + margin)

  whereas:
    * d_ap: measured distance between anchor and positive features
    * d_an: measured distance between anchor and negative features
    * margin: a hyperameter (think SVM)

  Args:
   margin:

  Returns:
    triplet_loss - tf.float32 scalar
  """
  def compute(y_true, y_pred):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    distance_ap = tf.reduce_sum(tf.square(anchor - positive), 1)
    distance_an = tf.reduce_sum(tf.square(anchor - negative), 1)

    triplet_loss = tf.maximum(0.0, distance_ap - distance_an + margin)
    triplet_loss = tf.reduce_mean(triplet_loss)

    # @TODO: how to deal with numerical instability?

    return triplet_loss

  return compute