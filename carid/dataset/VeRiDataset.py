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

  def __init__(self, root_dir, preprocess_func):
    self.preprocess_func = preprocess_func
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

  def _parse_record(self, record, mode):

    def read_resize(filename):
      image_string = tf.read_file(filename)
      image = tf.image.decode_image(image_string, channels=3)
      image = tf.to_float(image)
      image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
      image = self.preprocess(image, mode)
      return image

    features = tf.map_fn(
        fn=read_resize,
        elems=record,
        dtype=tf.float32)

    if mode == tf.estimator.ModeKeys.PREDICT:
      return {'anchor': features[0]}
    else:
      return {
          'anchor': features[0],
          'positive': features[1],
          'negative': features[2]}

  def preprocess(self, image, mode):
    """Preprocessor for VeriDataset
    Args:
      image: a Tensor - shape [height, width, channel]
      mode: `tf.estimator.ModeKeys`
    Returns:
    """
    return self.preprocess_func(image, mode)

  def get_input_fn(self,
                   mode,
                   data,
                   batch_size,
                   shuffle_buffer=200,
                   num_parallel_calls=4):
    """Create input_fn"""
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: self.generator(data_frames=data, mode=mode),
        output_types=tf.string,
        output_shapes=(tf.TensorShape([None])))
    dataset = dataset.prefetch(buffer_size=batch_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # dataset = dataset.repeat()
    dataset = dataset.map(
      map_func=lambda record: self._parse_record(record, mode),
      num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

  def batch_hard_generator(self, data_frames, mode):

    groups = data_frames.groupby('vehicleID')
    groups_name = groups.groups.keys()
    data_dir = os.path.join(self.root_dir, VeRiDataset.TRAIN_DIR)

    # Sample P classes, each class contains K samples.
    ids = random.sample(groups_name, 10)

    for id in ids:
      instances = groups.get_group(id).sample(30, replace=True)
      instances = [os.path.join(data_dir, i['imageName'])
                   for i in instances.to_dict('record')]

  def generator(self, data_frames, mode):
    """Generate a triple instances.

    Given a list of instances with different object IDs,
    this method would generate a triplet (anchor, positive, negative)
    instances for every iteration such that:

    * Anchor and positive instances have the same Object ID.
    * Negative instance have a object_id different than anchor, positive

    Args:
      df - pandas.DataFrame
      mode:

    Returns:
      generator - a data generator.

    """
    groups = data_frames.groupby('vehicleID')
    group_names = groups.groups.keys()
    data_dir = os.path.join(self.root_dir, VeRiDataset.TRAIN_DIR)

    def training_generator():
      while True:
        # randomly sample two groups
        ids = random.sample(group_names, 2)
        anchor, positive = groups.get_group(ids[0]).sample(2, replace=True).to_dict('records')
        negative = groups.get_group(ids[1]).sample(1).to_dict('records')[0]
        anchor, positive, negative = [
            os.path.join(data_dir, sample['imageName'])
            for sample in [anchor, positive, negative]]

        yield [anchor, positive, negative]

    def inference_generator():
      for sample in data_frames.to_dict(orient='records'):
        anchor = os.path.join(data_dir, sample['imageName'])

        yield [anchor]

    if mode == tf.estimator.ModeKeys.PREDICT:
      return inference_generator()
    else:
      return training_generator()
