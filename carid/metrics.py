import tensorflow as tf


def ap_distance(y_true, y_pred):
  anchor, positive = y_pred[:, 0, :, :], y_pred[:, 1, :, :]
  return tf.reduce_sum(tf.square(anchor - positive), 1)


def an_distance(y_true, y_pred):
  anchor, negative = y_pred[:, 0, :, :], y_pred[:, 2, :, :]
  return tf.reduce_sum(tf.square(anchor - negative), 1)
