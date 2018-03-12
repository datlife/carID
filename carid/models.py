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

  outputs = []
  for name in features:
    features[name].set_shape(shape=(None, None, None, 3))
    input_tensor = tf.keras.Input(tensor=features[name])

    with tf.variable_scope('car_id', reuse=tf.AUTO_REUSE):
      model = tf.keras.applications.ResNet50(
          input_tensor=input_tensor,
          include_top=False,
          pooling='avg',
          weights=None)

    outputs.append(model(features[name]))

  positive, anchor, negative = outputs
  # Compute Triplet Loss
  loss = params['loss_function'](anchor, positive, negative, params['margin'])
  # Create predictions and metrics
  predictions = {
      'anchor': anchor,
      'positive': positive,
      'negative': negative}

  metrics = {
      'ap_distance': tf.metrics.mean_squared_error(anchor, positive),
      'an_distance': tf.metrics.mean_squared_error(anchor, negative)}

  tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)
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

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_ops,
      eval_metric_ops=metrics)

