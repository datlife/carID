r"""An example of running carID on a video,
given provided list of car detections

"""
from carid.model import resnet

model = resnet()
model.summary()