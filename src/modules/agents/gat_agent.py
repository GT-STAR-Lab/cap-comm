import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATAgent(nn.Module):
    def __init__(self, input_shape, args, dropout=0.0, alpha=0.2, training=True):
        """Dense version of GAT."""
        super(GATAgent, self).__init__()
        self.args = args
        self.dropout = dropout
        msg_hidden_dim = args.msg_hidden_dim
        n_heads = self.args.n_heads
        n_actions = self.args.n_actions
        self.message_passes = self.args.num_layers
        
        self.encoder = nn.Sequential(nn.Linear(input_shape,self.args.hidden_dim),
                                      nn.ReLU(inplace=True))

        self.attentions = [GraphAttentionLayer(self.args.hidden_dim, msg_hidden_dim, args, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(msg_hidden_dim * n_heads, n_actions, args, dropout=dropout, alpha=alpha, concat=False)

        self.policy_head = nn.Sequential(nn.Linear(msg_hidden_dim * n_heads if self.message_passes > 0 else self.args.hidden_dim, self.args.hidden_dim),
                                         nn.ReLU(inplace=True))
        self.actions = nn.Linear(self.args.hidden_dim, n_actions)
    
    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.args.hidden_dim) #self.encoder.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, x, adj):
        B, N, N = adj.shape
        # print(x.shape)
        x = self.encoder(x)
        x = x.view(B, N, x.shape[1])
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        x = x.view(-1, x.shape[-1])
        h = self.policy_head(x)
        actions = self.actions(h)
        return actions, h

class DualChannelGATAgent(torch.nn.Module):
    """
    Dual Channel gnn (Agents that have two types of observations which they want to learn to communicate separately)
    """
    def __init__(self, input_shape, args, training=True):
        """
        """
        super(DualChannelGATAgent, self).__init__()
        self.args = args
        self.capability_shape = self.args.capability_shape
        self.input_shape_a = input_shape - self.capability_shape
        self.channel_A = GATAgent(self.input_shape_a, args=args, training=training)
        self.channel_B = GATAgent(self.capability_shape, args=args, training=training)

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
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, args, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.args = args

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        Batching formulation found here: https://github.com/Diego999/pyGAT/issues/36
        """
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        
        # (batch_zize, number_nodes, number_nodes, 2 * out_features)
        a_input = self.batch_prepare_attentional_mechanism_input(Wh)

        # (batch_zize, number_nodes, number_nodes, 2 * out_features) * (2 * out_features, 1)
        # -> (batch_zize, number_nodes, number_nodes, 1)
        e = torch.matmul(a_input, self.a)

        # (batch_zize, number_nodes, number_nodes)
        e = self.leakyrelu(e.squeeze(-1))

        # (batch_zize, number_nodes, number_nodes)
        zero_vec = -9e15 * torch.ones_like(e)

        # (batch_zize, number_nodes, number_nodes)
        attention = torch.where(adj > 0, e, zero_vec)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.softmax(attention, dim=-1)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # batched matrix multiplication (batch_zize, number_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def batch_prepare_attentional_mechanism_input(self, Wh):
        """
        with batch training
        :param Wh: (batch_zize, number_nodes, out_features)
        :return:
        """
        B, M, E = Wh.shape # (batch_zize, number_nodes, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)  # (B, M*M, E)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)  # (B, M*M, E)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)  # (B, M*M,2E)
        return all_combinations_matrix.view(B, M, M, 2 * E)

    def _prepare_attentional_mechanism_input(self, Wh_n):
        """
        no batch dimension
        :param Wh_n:
        :return:
        """
        M = Wh_n.size()[0]  # number of nodes（M, E)
        Wh_repeated_in_chunks = Wh_n.repeat_interleave(M, dim=0)  # (M, M, E)
        Wh_repeated_alternating = Wh_n.repeat(M, 1)  # (M, M, E)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)  # (M*M,2E)
        return all_combinations_matrix.view(M, M, 2 * self.out_features)  # （M, M, 2E)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

