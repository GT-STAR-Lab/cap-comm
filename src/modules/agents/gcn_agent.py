import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from modules.encoder import REGISTRY as encoder_REGISTRY
class GCNConv(MessagePassing):
    """"
    Graph Convolutional Neural Network defined in the tutorial 
    https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
    """
    def __init__(self, in_channels, out_channels):
        """
        params:
            in_channels (int): The feature vector length for Conv layer
        """
        super().__init__(aggr="add") # "Add" aggregated messages
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()


    def forward(self, x, edge_index):
        """
        params:
            x (tensor): Shape [N, in_channels]. 
            edge_index (tensor) : Shape [2, E]. Defines the graph topology
        """

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Step 2: Linearly transform node feature natrix
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages. Propagate will internally call message, aggregate, and update
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector
        out += self.bias

        return(out)
    
    def message(self, x_j, norm):
        """
        """
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCN(nn.Module):
    """2 layer graph convolution network"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class GCNAgent(nn.Module):
    """
    """
    def __init__(self, in_channels, args, training=True):
        super(GCNAgent, self).__init__()
        self.args = args

        self.in_channels = in_channels
        
        # encoder network
        self.encoder = encoder_REGISTRY["mlp"](in_channels, self.args.hidden_dim, self.args.hidden_dim)

        # two-layer gcn network
        self.gcn = GCN(in_channels=self.args.hidden_dim, hidden_channels=self.args.msg_hidden_dim, out_channels=self.args.msg_hidden_dim)

        self.actions = nn.Linear(self.args.msg_hidden_dim, self.args.n_actions)
    
    def init_hidden(self): #TODO: Where and why is this used?
        # make hidden states on same device as model
        return torch.zeros(self.args.hidden_dim) #self.encoder.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, x, adj_matrix):
        """
        params:
            The input shape should be [batch_size, N, input_size]
        """
        # convert adj_matrix to edge index (with batch size)
        edge_index = torch.nonzero(torch.squeeze(adj_matrix, dim=0), as_tuple=False).t()
        
        x = self.encoder(x)

        out = self.gcn(x, edge_index)

        actions = self.actions(out)
        return(actions, out)
        