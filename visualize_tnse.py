r"""Visualize t-SNE on trained car-ID model using Tensorboard

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf

from carid.losses import triplet_loss
from carid.dataset import VeRiDataset

