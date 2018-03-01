r"""Dataset loader for Triplet Model

"""
import os
import cv2
import pandas as pd
from lxml import etree
import tensorflow as tf


class DataLoader(object):
  """Base DataLoader

  """


class VeRiDataset(object):
  """VeRi Dataset Loader

  This dataset is for training vehicle re-identification model
  """
  FIELDS = ['vehicleID', 'imageName', 'typeID', 'cameraID']

  def __init__(self, path="VeRi"):
    self.path = path
    self.data = None

  def load(self):
    if not os.path.isfile(self.path):
      raise IOError("Cannot load %s" % self.path)
    with open(self.path, 'rb') as fio:
      xml_stream = fio.read()

    et = etree.fromstring(xml_stream)
    items = et.xpath('Items/Item')
    self.data = pd.DataFrame(
      [dict(item.attrib) for item in items],
      columns=VeRiDataset.FIELDS)
    return self

  def get_input_fn(self, is_training, batch_size, shuffle):
    """Create a input function

    Args:
      is_training:
      batch_size:
      shuffle:

    Returns:

    """

    if self.data is None:
      raise ValueError("Data is currently empty. Did you call load()?")

    dataset = tf.data.Dataset.from_tensor_slices(
      ({'input_anchor': None,
        'input_positive': None,
        'input_negative': None},
       [0.0, 0.0, 0.0])
    )

    dataset = dataset.shuffle(1000)
    dataset = dataset.make_one_shot_iterator().get_next()

    return dataset
