import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNComm(torch.nn.Module):
    def __init__(self, input_shape, args, training=True):
        super(GNNComm, self).__init__()
        self.args = args

        self.training = training

        self.encoder = nn.Sequential(nn.Linear(input_shape,self.args.hidden_dim),
                                      nn.ReLU(inplace=True))
        if self.args.use_graph_attention:
            self.messages = MultiHeadAttention(n_heads=self.args.n_heads,input_dim=self.args.hidden_dim,embed_dim=self.embed_dim)
        else:
            self.messages = nn.Sequential(nn.Linear(self.args.msg_hidden_dim,self.args.msg_hidden_dim, bias=False),
                                     nn.ReLU(inplace=True))


        self.policy_head = nn.Sequential(nn.Linear(self.args.msg_hidden_dim, self.args.hidden_dim),
                                         nn.ReLU(inplace=True))

        self.actions = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.n_actions),
                                         nn.ReLU(inplace=True))


    def cuda_transfer(self):
        for i in range(self.args.num_layers):
            self.convs[i].cuda()
            print('\n\n\n\n\nTRANFERED TO GPUUUU')

    def calc_adjacency_hat(self, adj_matrix):
        A_hat = adj_matrix + torch.eye(adj_matrix.shape[0]).to(self.device)
        D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        return torch.mm(torch.mm(D_hat, A_hat), D_hat)

    def forward(self,x,  batch_size, adj_matrix):
        # inp should be (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}
       
        comm_mat = self.calc_adjacency_hat(adj_matrix)
        comm_mat = comm_mat.unsqueeze(0).repeat(batch_size,1,1)
        # comm_mat - {batch_size, N, N}


        h = self.encoder(x) # should be (batch_size,self.hidden_dim)

        h = h.view(self.num_agents,-1,self.args.hidden_dim).transpose(0,1) # should be (batch_size/N,N,self.args.hidden_dim)
        
        for k in range(self.K):
            # m, attn = self.messages(h, mask=mask, return_attn=True) # should be <batch_size/N,N,self.embed_dim>
            h = self.messages(torch.mm(comm_mat,h))
        #h = h.transpose(0,1).contiguous().view(-1,self.args.hidden_dim)

        h = self.policy_head(h)
        actions = self.actions(h)
        
        return  actions, h