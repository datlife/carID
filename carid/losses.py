r""" Triplet Loss Function"""
import tensorflow as tf

def triplet_loss(anchor, positive, negative, margin=0.2):
  """Triplet Loss Implementation

  TripletLoss = Max(0.0, d_ap - d_an + margin)

  whereas:
    * d_ap: measured distance between anchor and positive features
    * d_an: measured distance between anchor and negative features
    * margin: a hyper-parameter (think SVM)
  Args:
   margin: float

  Returns:
    triplet_loss - tf.float32 scalar
  """
  distance_ap = tf.reduce_sum(tf.square(anchor - positive), 1)
  distance_an = tf.reduce_sum(tf.square(anchor - negative), 1)

  loss = tf.maximum(0.0, distance_ap - distance_an + margin)
  loss = tf.reduce_mean(loss)

  return loss

