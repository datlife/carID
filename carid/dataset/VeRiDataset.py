import os
import random
import pandas as pd
from lxml import etree

from DataProvider import DataProvider
import tensorflow as tf

class VeRiDataset(DataProvider):
  """Data Provider for VeRi, a vehicle re-identification Dataset.

  Source : https://github.com/VehicleReId/VeRidataset

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
    """Parse XML and load data as `pd.DataFrames` into `self.data`"""

    xml_file = os.path.join(self.root_dir, VeRiDataset.TRAIN_XML)
    if not os.path.isfile(xml_file):
      raise IOError("Cannot load %s" % xml_file)
    with open(xml_file, 'rb') as fio:
      xml_stream = fio.read()
    et = etree.fromstring(xml_stream)
    items = et.xpath('Items/Item')
    self.data = pd.DataFrame(
        [dict(item.attrib) for item in items],
        columns=VeRiDataset.FIELDS)
    print('Loaded %s training instances from VeRi Dataset' % len(self.data))
    return self

  def get_input_fn(self,
                   mode,
                   dataset,
                   batch_size,
                   parse_fn,
                   steps_per_epoch,
                   shuffle_buffer=200,
                   num_parallel_calls=4):
    """

    Args:
      mode:
      dataset:
      batch_size:
      parse_fn:
      steps_per_epoch:
      shuffle_buffer:
      num_parallel_calls:

    Returns:

    """
    train_eval_gen = self.batch_hard_generator(
        data_frames=dataset,
        mode=mode,
        batch_size=batch_size,
        samples_per_class=8)
    training_instances = [next(train_eval_gen) for _ in range(steps_per_epoch)]
    paths, labels = zip(*training_instances)
    paths = [i for p in paths for i in p]
    labels = [i for l in labels for i in l]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.map(
        lambda filename, label: parse_fn(filename, label, mode),
        num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

  def batch_hard_generator(self, data_frames, mode, batch_size, samples_per_class=8):
    """ Sample P classes, each class contains K samples.  """

    data_dir = os.path.join(self.root_dir, VeRiDataset.TRAIN_DIR)
    data_frames['imageName'] = data_frames['imageName'].apply(
        lambda i: os.path.join(data_dir, i))

    groups = data_frames.groupby('vehicleID')
    groups_name = groups.groups.keys()
    num_classes = batch_size // samples_per_class

    while True:
      ids = random.sample(groups_name, num_classes)
      df = pd.concat([groups.get_group(idx).sample(
          samples_per_class, replace=True)
          for idx in ids])

      yield list(df['imageName']), list(df['vehicleID'])

