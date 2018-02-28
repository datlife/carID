r"""Using a pre-trained Resnet-50

"""
import tensorflow as tf


def resnet(include_top=False, input_shape=(None, None, 3)):
  """Download (if needed) and construct pretrained ResNet-50 model

  """
  model = tf.keras.applications.resnet50.ResNet50(
    include_top=include_top,
    input_shape=input_shape)

  return model

