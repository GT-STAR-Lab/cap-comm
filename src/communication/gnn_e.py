import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

class GCNVanilla(torch.nn.Module):
    def __init__(self, input_shape, args, training=True):
        super(GCNComm, self).__init__()
        self.args = args
        self.training = training
        self.convs = []
        print(input_shape)
        print(self.args.msg_hidden_dim)

        self.input_encoder = nn.Linear(input_shape, args.hidden_dim)
        self.messages = nn.Linear(args.hidden_dim, args.msg_hidden_dim)
        self.convs.append(nn.Sequential(nn.Linear(input_shape,self.h_dim),
                                     self.nonlin(inplace=True))) 

        for i in range(1, self.args.num_layers-1):
            self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_hidden_dim).to())

        self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_out_size))

    def cuda_transfer(self):
        for i in range(self.args.num_layers):
            self.convs[i].cuda()
            print('\n\n\n\n\nTRANFERED TO GPUUUU')

    def forward(self, x, adj_matrix):
        x_out = []

        for x_in, am_in in zip(torch.unbind(x, dim=0), torch.unbind(adj_matrix, dim=0)):
            print(x_in.shape)
            for i in range(self.args.num_layers):
                x_in = self.convs[i](x_in, dense_to_sparse(am_in)[0])
                if (i+1)<self.args.num_layers:
                    x_in = F.elu(x_in) #DESIGN CHOICE
                    x_in = F.dropout(x_in, p=0.2, training=self.training) # DESIGN CHOICE
            x_out.append(x_in)
        return torch.stack(x_out, dim=0)
