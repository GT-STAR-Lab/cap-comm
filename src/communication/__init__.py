from .gcn import GCNComm, GATComm
from .gnn import GNNComm
REGISTRY = {}

REGISTRY["gcn"] = GCNComm
REGISTRY["gat"] = GATComm
REGISTRY["gnn"] = GNNComm