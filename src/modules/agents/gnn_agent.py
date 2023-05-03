import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

class GNNAgent(torch.nn.Module):
    def __init__(self, input_shape, args, training=True):
        super(GNNAgent, self).__init__()
        self.args = args

        self.training = training
        self.input_shape = input_shape
        self.message_passes = self.args.num_layers
    
        self.encoder = nn.Sequential(nn.Linear(input_shape,self.args.hidden_dim),
                                      nn.ReLU(inplace=True))
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
        A_hat = (adj_matrix + torch.eye(adj_matrix.shape[-1])).squeeze()#.to(self.device)
        D_hat = torch.pow(A_hat.sum(1), -0.5).unsqueeze(-1) * torch.ones(A_hat.shape)

        return torch.matmul(torch.matmul(D_hat, A_hat), D_hat)

    def forward(self,x, adj_matrix):
        # inp should be (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}


        # print(x[:4, 0:3])        
        comm_mat = self.calc_adjacency_hat(adj_matrix)
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
        
        
        msg = msg.view(-1, self.args.hidden_dim) # shape is (batch_size * N, self.args.hidden_dim)
        if self.message_passes > 0:
            h = torch.cat((enc,msg), dim=-1)
        else:
            h = enc

        h = self.policy_head(h)
        actions = self.actions(h)

    
        
        return  actions, h