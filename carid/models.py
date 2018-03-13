"""Construct a CarID model"""
import tensorflow as tf

def resnet_carid(multi_gpu):
  func = resnet50_model_fn if not multi_gpu else \
      tf.contrib.estimator.replicate_model_fn(
         resnet50_model_fn, tf.losses.Reduction.MEAN)
  return func


def resnet50_model_fn(features, labels, mode, params):
  """Model Function for tf.estimator.Estimator object

  Note that because of triplet loss function, we do not need labels
  """
  # Determine if model should update weights
  tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)
  outputs = {}
  for name in features:
    features[name].set_shape((None, None, None, 3))
    with tf.variable_scope('car_id', reuse=tf.AUTO_REUSE):
      model = tf.keras.applications.ResNet50(
          input_tensor=tf.keras.Input(tensor=features[name]),
          include_top=False,
          weights=None)
    outputs[name] = model(features[name])
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'anchor': outputs['anchor']}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  positive, anchor, negative = [outputs[i]
                                for i in ['anchor', 'positive', 'negative']]
  # compute triplet loss
  distance_ap = tf.reduce_sum(tf.square(anchor - positive), 1)
  distance_an = tf.reduce_sum(tf.square(anchor - negative), 1)
  loss = tf.maximum(0.0, distance_ap - distance_an + params['margin'])
  loss = tf.reduce_mean(loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    optimizer = params['optimizer'](params['learning_rate'])
    if params['multi_gpu']:
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    # for batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_ops = optimizer.minimize(loss, global_step)
  else:
    train_ops = None

  predictions = {
    'anchor': anchor,
    'positive': positive,
    'negative': negative}

  tf.identity(loss, 'train_loss')
  tf.summary.scalar('train_loss', loss)

  ap_dist = tf.reduce_mean(distance_ap)
  tf.identity(ap_dist, 'train_ap_dist')
  tf.summary.scalar('train_ap_dist', ap_dist)

  an_dist = tf.reduce_mean(distance_an)
  tf.identity(an_dist, 'train_an_dist')
  tf.summary.scalar('train_an_dist', an_dist)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_ops,
      eval_metric_ops={})

