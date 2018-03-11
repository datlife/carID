r"""Dataset loader for Triplet Model

"""
import os
import cv2
import abc
import random
import pandas as pd

from lxml import etree
import tensorflow as tf
import multiprocessing as mp
from sklearn.model_selection import train_test_split


class DataProvider(object):
  """"Abstract base class for Dataset object
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, root_dir):
    """Constructor

    Args:
      root_dir: - String - absolute path
        to dataset directory. It should contain all necessary training/testing
        instances and labels.
    """
    self.root_dir = root_dir
    self.data = None

  @abc.abstractmethod
  def load(self):
    """"Initialize `self.data` for other methods to use"""
    pass

  @abc.abstractmethod
  def preprocess(self, mode):
    """Preprocess function for inputs"""
    pass

  @abc.abstractmethod
  def _parse_record(self, record, mode):
    """Parse data record into training instance

    It is usually helpful for large dataset. A record might contain filenames
    and labels, and during constructing dataset. This allows us to take
    advantage of parallel processing tf.Dataset, instead of consecutively
    reading images for disk.


    Args:
      record:
      mode:

    Returns:

    """

  @abc.abstractmethod
  def generator(self, data_frames, mode):
    """Determine how to generate a single instance (features, label) for
    training/evaluation

    Returns:
      a generator - yields a single instance (features, label)
    """
    pass

  def get_input_fn(self,
                   mode,
                   data,
                   epochs,
                   batch_size,
                   shuffle_buffer=200,
                   num_parallel_calls=4):
    """Create a input function

    Args:
      mode:
      data:
      epochs:
      batch_size:
      shuffle_buffer:
      num_parallel_calls:

    Returns:

    """
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: self.generator(data_frames=data),
        output_types=3 * (tf.float32,),
        output_shapes=3 * (tf.TensorShape([None, None, 3]),))

    dataset = dataset.prefetch(buffer_size=batch_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.repeat(epochs)
    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(
          lambda image: self.preprocess(mode),
          num_parallel_calls=num_parallel_calls)
    else:
      dataset = dataset.map(
          lambda image, label: self.preprocess(mode),
          num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

  def split_training_data(self, test_size, shuffle=False):
    if self.data is None:
      raise ValueError("Data is currently empty. Did you call load()?")

    training_data, validation_data = train_test_split(
        self.data,
        test_size=test_size,
        shuffle=shuffle)

    return training_data, validation_data


class VeRiDataset(DataProvider):
  """Data Provider for VeRi, a vehicle re-identification Dataset.


  Attributes:
    FIELDS - a list of headers
    TRAIN_XML - training label file
    TRAIN_DIR - path to images for training

  """
  TRAIN_XML = 'train_label.xml'
  TRAIN_DIR = 'image_train'
  FIELDS = ['vehicleID', 'imageName', 'typeID', 'cameraID']

  def __init__(self, root_dir):
    super(VeRiDataset, self).__init__(root_dir)

  def load(self):
    """Parse XML and load data as `pd.DataFrames` into `self.data`
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

  def preprocess(self, mode):
    """Preprocessor for VeriDataset

    Args:
      mode: `tf.estimator.ModeKeys`

    Returns:

    """
    def convert_to_dict(anchor, positive, negative):
      return ({
        'anchor':   anchor,
        'positive': positive,
        'negative': negative,
      }, tf.random_normal([1, 1, 3]))  # dummy labels

    return convert_to_dict

  def generator(self, data_frames, mode):
    """Generate a triple instances.

    Given a list of instances with different object IDs,
    this method would generate a triplet (anchor, positive, negative)
    instances for every iteration such that:

    * Anchor and positive instances have the same Object ID.
    * Negative instance have a object_id different than anchor, positive

    Args:
      data_frames - pandas.DataFrame
      mode:

    Returns:
      generator - a data generator.

    """
    groups = data_frames.groupby('vehicleID')
    group_names = groups.groups.keys()
    data_dir = os.path.join(self.root_dir, VeRiDataset.TRAIN_DIR)

    def training_generator():
      # randomly sample two groups
      ids = random.sample(group_names, 2)
      anchor, positive = groups.get_group(ids[0]).sample(2).to_dict('records')
      negative = groups.get_group(ids[1]).sample(1).to_dict('records')[0]
      anchor, positive, negative = [
        os.path.join(data_dir, sample['imageName'])
        for sample in [anchor, positive, negative]]

      yield anchor, positive, negative

    def inference_generator():
      for row in data_frames.iterrows():
        record = row.to_dict('record')
        yield os.path.join(data_dir, record['imageName'])

    while True:
      if mode == tf.estimator.ModeKeys.PREDICT:
        inference_generator()
      else:
        training_generator()


