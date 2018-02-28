r""" Triplet Loss Function

This file contains the definition of Triplet Loss

"""
import tensorflow as tf


def triplet_loss(anchor, positve, negative, margin=0.2):
  """Definition of Triplet Loss


  Args:
    anchor:
    positve:
    negative:

  Returns:

  """
  distance_ap = tf.reduce_sum(tf.square(anchor - positive), 1)
  distance_an = tf.reduce_sum(tf.square(anchor - negative), 1)

  loss = tf.maximum(0, distance_ap - distance_an + margin)
  loss = tf.reduce_mean(loss)

  return loss
