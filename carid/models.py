"""Construct a CarID model"""
import tensorflow as tf


def resnet_carid(multi_gpu):
  func = resnet50_model_fn if not multi_gpu else \
      tf.contrib.estimator.replicate_model_fn(
         resnet50_model_fn, tf.losses.Reduction.MEAN)
  return func


def resnet50_model_fn(features, labels, mode, params):
  """Construct ResNet model_fn

  Args:
    features:
    labels:
    mode:
    params:

  Returns:

  """
  global _INIT_WEIGHTS

  # Determine if model should update weights
  tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)

  model = tf.keras.applications.ResNet50(
      input_tensor=tf.keras.Input(tensor=features),
      include_top=False,
      pooling='avg',
      weights=None)

  avg_pool = model(features)
  embeddings = tf.keras.layers.Dense(128, activation=None)(avg_pool)

  loss, distance_ap, distance_an, num_active = params['loss_fn'](
    embeddings, labels, params['margin'])

  # L2 regularization
  l2_term = tf.add_n([tf.nn.l2_loss(t) for t in tf.trainable_variables()])
  loss = loss + params['weight_decay'] * l2_term

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    optimizer = params['optimizer']
    if params['multi_gpu']:
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    # for batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_ops = optimizer.minimize(loss, global_step)

    # Add Summary
    ap_dist = tf.reduce_mean(distance_ap)
    an_dist = tf.reduce_mean(distance_an)
    tf.identity(ap_dist, 'ap_dist')
    tf.summary.scalar('ap_dist', ap_dist)
    tf.identity(an_dist, 'an_dist')
    tf.summary.scalar('an_dist', an_dist)
    tf.summary.scalar('num_active', num_active)
  else:
    train_ops = None

  tf.summary.image('images', features, max_outputs=6)
  tf.identity(loss, 'loss')
  tf.summary.scalar('loss', loss)

  metrics = {
    'ap_distance_metric': tf.metrics.mean(distance_ap),
    'an_distance_metric': tf.metrics.mean(distance_an)}
  predictions = {'embeddings': embeddings}

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_ops,
      eval_metric_ops=metrics)

