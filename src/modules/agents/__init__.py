REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .mlp_agent import MLPAgent
from .gnn_agent import GNNAgent, DualChannelGNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["gnn"] = GNNAgent
REGISTRY["dual_gnn"] = DualChannelGNNAgent
