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
    """"Initialize `self.data` for other methods to use
    """
    pass

  @abc.abstractmethod
  def preprocess(self, is_training):
    """Define how to preprocess a single instance for
    training/evaluation.

    Args:
      is_training: a boolean
      preprocess_func: a callable function

    Returns:
      transformed features
    """
    pass

  @abc.abstractmethod
  def generator(self, data_frames):
    """Determine how to generate a single instance (features, label) for
    training/evaluation

    Returns:
      a generator - yields a single instance (features, label)
    """
    pass

  def get_input_fn(self,
                   is_training,
                   batch_size,
                   shuffle,
                   buffer_size=200,
                   num_threads=4):
    """Create a input function

    Args:
      is_training: boolean
      batch_size:  int
      shuffle:     boolean
      buffer_size: int
      num_threads: int

    Returns:

    """
    if self.data is None:
      raise ValueError("Data is currently empty. Did you call load()?")

    training_data, validation_data = train_test_split(
      self.data,
      test_size=0.3,
      shuffle=True)

    # @TODO : create Transformer
    input_funcs = [(
      tf.data.Dataset.from_generator(
        lambda: self.generator(data_frames=data_frames),
        3 * (tf.float32,),
        3 * (tf.TensorShape([None, None, 3]),)).
      repeat().
      prefetch(buffer_size).
      map(self.preprocess(is_training), num_parallel_calls=mp.cpu_count()).
      prefetch(buffer_size).
      shuffle(buffer_size).
      batch(batch_size).
      make_one_shot_iterator().
      get_next())
      for data_frames in [training_data, validation_data]]

    return input_funcs


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

  def preprocess(self, is_training):
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
      }, tf.random_normal([1, 1, 3]))  # dummy labels

    return convert_to_dict

  def generator(self, data_frames):
    """Generate a triple instances.

    Given a list of instances with different object IDs,
    this method would generate a triplet (anchor, positive, negative)
    instances for every iteration such that:

    * Anchor and positive instances have the same Object ID.
    * Negative instance have a object_id different than anchor, positive

    Args:
      data_frames - pandas.DataFrame

    Returns:
      generator - a data generator.

    """

    groups = data_frames.groupby('vehicleID')
    group_names = groups.groups.keys()

    data_dir = os.path.join(self.root_dir, VeRiDataset.TRAIN_DIR)
    while True:
      ids = random.sample(group_names, 2)
      anchor, positive = groups.get_group(ids[0]).sample(2).to_dict('records')
      negative = groups.get_group(ids[1]).sample(1).to_dict('records')[0]

      anchor, positive, negative = [
        cv2.resize(cv2.imread(
          os.path.join(data_dir, sample['imageName']),
          cv2.IMREAD_COLOR),
          (224, 224))
        for sample in [anchor, positive, negative]]

      yield anchor, positive, negative

  def get_samples(self, per_class_samples=50, num_classes=None):
    """Load

    Args:
      per_class_samples:
      num_classes:

    Returns:

    """
    groups = self.data.groupby('vehicleID')
    groups_names = groups.groups.keys()

    selected_classes = random.sample(groups_names, num_classes)
    data_dir = os.path.join(self.root_dir, VeRiDataset.TRAIN_DIR)

    samples = {}
    for cls in selected_classes:
      samples[cls] = groups.get_group(cls).sample(
        per_class_samples, replace=True).to_dict('records')
    return samples
