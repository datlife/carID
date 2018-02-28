r""" Triplet Loss Function

This file contains the definition of Triplet Loss

"""
import tensorflow as tf


def triplet_loss(anchor, positve, negative):
  """Definition of Triplet Loss


  Args:
    anchor:
    positve:
    negative:

  Returns:

  """
  margin = 0.2
  distance_ap = None
  distance_an = None

  loss = tf.maximum(0, tf.pow(distance_ap, 2)-tf.pow(distance_an, 2)+margin)

  return loss
