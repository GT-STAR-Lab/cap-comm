import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
import sys

class GNNAgent(torch.nn.Module):
    def __init__(self, input_shape, args, training=True):
        super(GNNAgent, self).__init__()
        self.args = args

        self.training = training
        self.input_shape = input_shape
        self.message_passes = self.args.num_layers

        
        self.encoder = nn.Sequential(nn.Linear(input_shape,self.args.hidden_dim),
                                      nn.ReLU(inplace=True))
        # self.encoder = encoder_REGISTRY[self.args.encoder](input_shape, self.args.hidden_dim, self.args.hidden_dim)

        if self.args.use_graph_attention:
            self.messages = nn.MultiHeadAttention(n_heads=self.args.n_heads,input_dim=self.args.hidden_dim,embed_dim=self.embed_dim)
        else:
            self.messages = nn.Sequential(nn.Linear(self.args.msg_hidden_dim,self.args.msg_hidden_dim, bias=False),
                                     nn.ReLU(inplace=True))


        self.policy_head = nn.Sequential(nn.Linear(self.args.msg_hidden_dim + self.args.hidden_dim if self.message_passes > 0 else self.args.hidden_dim, self.args.hidden_dim),
                                         nn.ReLU(inplace=True))

        self.actions = nn.Linear(self.args.hidden_dim, self.args.n_actions)

    def cuda_transfer(self):
        for i in range(self.args.num_layers):
            self.convs[i].cuda()
            print('\n\n\n\n\nTRANFERED TO GPUUUU')

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.args.hidden_dim) #self.encoder.weight.new(1, self.args.hidden_dim).zero_()

    def calc_adjacency_hat(self, adj_matrix):
        """
        Calculates the normalized adjacency matrix including self-loops.
        This bounds the eigenv values and repeated applications of this graph
        shift operator could lead to numerical instability if this is not done, as
        well as exploding/vanishing gradients.
        """
        # This adds a self-loop so that a nodes own message is passed onto itself
        A_hat = (adj_matrix + torch.eye(adj_matrix.shape[-1])).squeeze()#.to(self.device) 

        #
        D_hat = torch.pow(A_hat.sum(1), -0.5).unsqueeze(-1) * torch.ones(A_hat.shape)

        return torch.matmul(torch.matmul(D_hat, A_hat), D_hat)

    def forward(self,x, adj_matrix):
        
        # inp should be (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}


        # Get the Normalized adjacency matrix, and add self-loops  
        if(self.args.normalize_adj_matrix):     
            comm_mat = self.calc_adjacency_hat(adj_matrix)
        else:
            comm_mat = adj_matrix.float()
        
        #comm_mat = comm_mat#.unsqueeze #(0).repeat(batch_size,1,1)
       
        # comm_mat - {batch_size, N, N}
        enc = self.encoder(x) # should be (batch_size,self.hidden_dim)

        msg = enc.view(-1, self.args.n_agents, self.args.hidden_dim) # shape is (batch_size, N, self.hidden_dim)

        # h = h.view(self.args.n_agents,-1,self.args.hidden_dim).transpose(0,1) # should be (batch_size/N,N,self.args.hidden_dim)
        
        for k in range(self.message_passes):
        # #     # m, attn = self.messages(h, mask=mask, return_attn=True) # should be <batch_size/N,N,self.embed_dim>
        # #     #print(comm_mat, h.shape)
        #     # print(comm_mat.shape, h.shape)
            msg = self.messages(torch.matmul(comm_mat,msg))
        #h = h.transpose(0,1).contiguous().view(-1,self.args.hidden_dim)
        
        
        # passess the agents embedded encoding to the policy head.
        msg = msg.view(-1, self.args.hidden_dim) # shape is (batch_size * N, self.args.hidden_dim)
        # print("Message shape", msg.shape, "Encoding shape", enc.shape)
        if self.message_passes > 0:
            h = torch.cat((enc,msg), dim=-1)
        else:
            h = enc

        h = self.policy_head(h)
        actions = self.actions(h)

    
        
        return  actions, h
    
class DualChannelGNNAgent(torch.nn.Module):
    """
    Dual Channel gnn (Agents that have two types of observations which they want to learn to communicate separately)
    """
    def __init__(self, input_shape, args, training=True):
        """
        """
        super(DualChannelGNNAgent, self).__init__()
        self.args = args
        self.capability_shape = self.args.capability_shape
        self.input_shape_a = input_shape - self.capability_shape
        self.channel_A = GNNAgent(self.input_shape_a, args=args, training=training)
        self.channel_B = GNNAgent(self.capability_shape, args=args, training=training)

        self.actions = nn.Linear(2*self.args.hidden_dim, self.args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        self.channel_A.init_hidden()
        self.channel_B.init_hidden()
        return torch.zeros(self.args.hidden_dim) #self.encoder.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, x, adj_matrix):
        """
        Forward the two inputs to each channel and adjacency matrix
        through the model

        params:
            x tensor
        returns:
            action (tensor)
            h (tensor) : concatenation of the two gnn outputs.
        """
        x_a = x[:, :self.input_shape_a]
        x_b = x[:, self.input_shape_a:] # should be the capabilities
        _, h_a = self.channel_A(x_a, adj_matrix)
        _, h_b = self.channel_B(x_b, adj_matrix)

        h = torch.concat((h_a, h_b), dim=-1)
        action = self.actions(h)
        return(action, h)