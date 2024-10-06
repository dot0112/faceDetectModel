import sys

sys.path.append(r"D:\mtcnn\pnet/training_pnet")
from training_pnet.training_pnet import training_pnet

training_pnet("wider", max_epochs=500)
