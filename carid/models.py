import tensorflow as tf


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
