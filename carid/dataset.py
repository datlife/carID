r"""Dataset loader for Triplet Model

"""
import os
import cv2
import random
import pandas as pd

from lxml import etree
import tensorflow as tf
import multiprocessing as mp
from sklearn.model_selection import train_test_split


class VeRiDataset(object):
  """VeRi Dataset Loader

  This dataset is for training vehicle re-identification model

  Default attributes

  FIELDS - a list of headers
  TRAIN_XML - training label file
  TRAIN_DIR - path to images for training

  """
  TRAIN_XML = 'train_label.xml'
  TRAIN_DIR = 'image_train'
  FIELDS = ['vehicleID', 'imageName', 'typeID', 'cameraID']

  def __init__(self, root_dir):
    self.data = None
    self.root_dir = root_dir

  def load(self):
    """Parse XML and load data into `self.data`
    """
    xml_file = os.path.join(self.root_dir, VeRiDataset.TRAIN_XML)

    if not os.path.isfile(xml_file):
      raise IOError("Cannot load %s" % path)

    with open(xml_file, 'rb') as fio:
      xml_stream = fio.read()
    et = etree.fromstring(xml_stream)

    items = et.xpath('Items/Item')
    self.data = pd.DataFrame(
      [dict(item.attrib) for item in items],
      columns=VeRiDataset.FIELDS)

    return self

  def get_input_fn(self, is_training, batch_size, shuffle, buffer_size=1000):
    """Create a input function

    Args:
      is_training:
      batch_size:
      shuffle:

    Returns:

    """

    if self.data is None:
      raise ValueError("Data is currently empty. Did you call load()?")

    training, validation = train_test_split(self.data, test_size=0.3)
    # @TODO : create Transformer

    training_data = tf.data.Dataset.from_generator(
      lambda: self.generator(
        data_frames=training,
        is_training=is_training),
      3 * (tf.float32,),
      3 * (tf.TensorShape([None, None, 3]),))

    training_data = (training_data.prefetch(buffer_size).
              map(self._preprocess(is_training),
                  num_parallel_calls=mp.cpu_count()).
              prefetch(buffer_size).
              shuffle(buffer_size).
              batch(batch_size).
              make_one_shot_iterator().
              get_next())

    validataion_data = tf.data.Dataset.from_generator(
      lambda: self.generator(
        data_frames=validation,
        is_training=False),
      3 * (tf.float32,),
      3 * (tf.TensorShape([None, None, 3]),))

    validataion_data = (validataion_data.prefetch(buffer_size).
              map(self._preprocess(is_training),
                  num_parallel_calls=mp.cpu_count()).
              prefetch(buffer_size).
              batch(batch_size).
              make_one_shot_iterator().
              get_next())

    return training_data, validataion_data

  def _preprocess(self, is_training):
    """Preprocessor for VeriDataset

    Args:
      is_training:

    Returns:

    """
    def convert_to_dict(anchor, positive, negative):
      return ({
        'input_anchor':   anchor,
        'input_positive': positive,
        'input_negative': negative,
      }, tf.random_normal([1, 1, 3]))

    return convert_to_dict

  def generator(self,data_frames, is_training):
    """Generate a triple instances.

    Given a list of instances with different object IDs,
    this method would generate a triplet (anchor, positive, negative)
    instances for every iteration such that:

    * Anchor and positive instances have the same Object ID.
    * Negative instance have a object_id different than anchor, positive

    Args:
      data_frames - pandas.DataFrame
      is_training -  Boolean


    Returns:
      generator - a data generator.

    """

    groups = data_frames.groupby('vehicleID')
    group_names = groups.groups.keys()

    data_dir = os.path.join(self.root_dir,VeRiDataset.TRAIN_DIR)
    while True:
      ids = random.sample(group_names, 2)
      anchor, positive = groups.get_group(ids[0]).sample(2).to_dict('records')
      negative = groups.get_group(ids[1]).sample(1).to_dict('records')[0]
      anchor, positive, negative = [
        cv2.resize(
          cv2.imread(
            os.path.join(data_dir, sample['imageName']),
            cv2.IMREAD_COLOR),
          (224, 224))
        for sample in [anchor, positive, negative]
      ]

      yield anchor, positive, negative