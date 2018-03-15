r"""Dataset loader for Triplet Model

"""

import abc
from sklearn.model_selection import train_test_split


class DataProvider(object):
  """"Abstract base class for Dataset object
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, root_dir):
    """Instantiate `DataProvider` instance

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
  def preprocess(self, input_data, mode):
    """Preprocess function for inputs"""
    pass

  @abc.abstractmethod
  def _parse_record(self, record, mode):
    """Parse data record into training instance

    It is usually helpful for large dataset. A record might contain meta-data
    such as filenames and labels. This allows us to take advantage of parallel
    processing tf.Dataset, instead of consecutively reading images from disk.

    Args:
      record:
      mode:
    Returns:
    """
    pass

  @abc.abstractmethod
  def generator(self, data_frames, mode):
    """Determine how to generate a single instance (features, label) for
    training/evaluation

    Returns:
      a generator - yields a single instance (features, label)
    """
    pass

  @abc.abstractmethod
  def get_input_fn(self,
                   mode,
                   data,
                   batch_size,
                   shuffle_buffer=200,
                   num_parallel_calls=4):
    """Create a input function"""
    pass

  def split_training_data(self, test_size, shuffle=False):
    if self.data is None:
      raise ValueError("Data is currently empty. Did you call load()?")
    training_data, validation_data = train_test_split(
        self.data,
        test_size=test_size,
        shuffle=shuffle)
    return training_data, validation_data


