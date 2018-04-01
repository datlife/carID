r""" Triplet Loss Function"""
import numbers
import tensorflow as tf


def triplet_loss(anchor, positive, negative, margin=0.2):
  """Vanilla Triplet Loss Implementation

  TripletLoss = Max(0.0, d_ap - d_an + margin)

  whereas:
    * d_ap: measured distance between anchor and positive embeddings
    * d_an: measured distance between anchor and negative embeddings
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


def batch_hard_triplet_loss(embeddings, pids, margin=0.2):
  """Computes the batch-hard loss from arxiv.org/abs/1703.07737.
  Args:
      embeddings (2D tensor): outputs from feature_extractor [B, num_features] .
      pids (1D tensor): The identities of the entries in `batch`, shape (B,).
          This can be of any type that can be compared, thus also a string.
      margin: The value of the margin if a number, alternatively the string
          'soft' for using the soft-margin formulation, or `None` for not
          using a margin at all.
  Returns:
      A 1D tensor of shape (B,) containing the loss value for each sample.
  """
  dists = _compute_pairwise_distance(embeddings, embeddings)
  same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                tf.expand_dims(pids, axis=0))

  negative_mask = tf.logical_not(same_identity_mask)
  positive_mask = tf.logical_xor(same_identity_mask,
                                 tf.eye(tf.shape(pids)[0], dtype=tf.bool))

  hardest_positive = tf.reduce_max(
      dists * tf.cast(positive_mask, tf.float32), axis=1)
  hardest_negative = tf.map_fn(
      lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
      (dists, negative_mask), tf.float32)

  losses = hardest_positive - hardest_negative

  if isinstance(margin, numbers.Real):
    losses = tf.maximum(margin + losses, 0.0)
  elif margin == 'soft':
    losses = tf.nn.softplus(losses)
  else:
    raise ValueError(
        'margin can be either a float or `soft`')

  num_active = tf.reduce_sum(tf.cast(tf.greater(losses, 1e-5), tf.float32))
  loss = tf.reduce_mean(losses)
  return loss, hardest_positive, hardest_negative, num_active


def _compute_pairwise_distance(a, b, metric='euclidean'):
  """Similar to scipy.spatial's cdist, but symbolic.
  The currently supported metrics can be listed as `cdist.supported_metrics`
  and are:
      - 'euclidean', although with a fudge-factor epsilon.
      - 'sqeuclidean', the squared euclidean.
      - 'cityblock', the manhattan or L1 distance.
  Args:
      a (2D tensor): The left-hand side, shaped (B1, F).
      b (2D tensor): The right-hand side, shaped (B2, F).
      metric (string): Which distance metric to use, see notes.
  Returns:
      The matrix of all pairwise distances between all vectors in `a` and in
      `b`, will be of shape (B1, B2).
  Note:
      When a square root is taken (such as in the Euclidean case), a small
      epsilon is added because the gradient of the square-root at zero is
      undefined. Thus, it will never return exact zero in these cases.
  """
  with tf.name_scope("cdist"):
    diffs = _all_diffs(a, b)
    if metric == 'sqeuclidean':
      return tf.reduce_sum(tf.square(diffs), axis=-1)
    elif metric == 'euclidean':
      return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
    elif metric == 'cityblock':
      return tf.reduce_sum(tf.abs(diffs), axis=-1)
    else:
      raise ValueError(
          'Unknown distance metrics')


def _all_diffs(a, b):
  """ Returns a tensor of all combinations of a - b.
  Args:
      a (2D tensor): A batch of vectors shaped (B1, F).
      b (2D tensor): A batch of vectors shaped (B2, F).
  Returns:
      The matrix of all pairwise differences between all vectors in `a` and in
      `b`, will be of shape (B1, B2).
  Note:
      For convenience, if either `a` or `b` is a `Distribution` object, its
      mean is used.
  """
  return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)
