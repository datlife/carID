"""Construct a CarID model"""
import tensorflow as tf

def carid_net(multi_gpu):
  func = model_fn if not multi_gpu else \
      tf.contrib.estimator.replicate_model_fn(
          model_fn, tf.losses.Reduction.MEAN)
  return func


def model_fn(features, labels, mode, params):
  """Model Function for tf.estimator.Estimator object
  Note that because of triplet loss training, we do not need labels
  """

  # Determine if model should update weights
  tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)

  model = tf.keras.applications.ResNet50(
      input_tensor=features,
      include_top=False)

  anchor = model(features['anchor'])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = model(features['anchor'])
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  positive = model(features['positive'])
  negative = model(features['negative'])

  # Compute Triplet Loss
  loss = params['loss_function'](anchor, positive, negative, params['margin'])

  # Create predictions and metrics
  predictions = {
      'anchor': anchor,
      'positive': positive,
      'negative': negative}

  metrics = {
    'ap_distance': tf.reduce_sum(tf.square(anchor - positive), 1),
    'an_distance': tf.reduce_sum(tf.square(anchor - negative), 1)}

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


def resnet50_triplet(input_shape=(224, 224, 3)):
  """Construct a Triplet CarID model

  This model takes in 3 inputs: [Anchor, Positive, Negative], which are
  images

  [Inputs] => [Feature Extractor (Resnet50)] ==> [Feature_map]

  Args:
    input_shape:

  Returns:

  """
  feature_extractor = tf.keras.applications.ResNet50(
    input_shape=input_shape,
    include_top=False, )

  inputs = [tf.keras.Input(
    shape=input_shape,
    name="input_%s" % in_type)
    for in_type in ['anchor', 'positive', 'negative']]

  outputs = tf.keras.layers.concatenate(
    inputs=[feature_extractor(features) for features in inputs],
    axis=1,
    name='output')

  carid = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="CarReId_net")

  return carid
