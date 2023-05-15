REGISTRY = {}

from .mlp_encoder import MLPEncoder
from .gru_encoder import GRUEncoder

REGISTRY["mlp"] = MLPEncoder
REGISTRY["gru"] = GRUEncoder