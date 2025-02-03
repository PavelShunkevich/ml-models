import os
import torch

from models.cifar10_pytorch.nn_modules import NetDropout
from torchviz import make_dot


# Creating a model instance
model = NetDropout(n_chans1=64)

# Creating dummy input
dummy_input = torch.randn(1, 3, 32, 32)  # Input size for CIFAR-10

# Add path to dot.exe (graphviz)
os.environ["PATH"] += os.pathsep + 'W:/graphviz/bin'  # change to your path

# Graph visualisation
dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
dot.render("net_dropout_graph", format="png") # generates net_dropout_graph.png file

print("Visualization generated in visualization/net_dropout_graph.png")