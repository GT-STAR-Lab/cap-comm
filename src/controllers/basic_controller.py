from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from communication import REGISTRY as comm_REGISTRY
from modules.critics import REGISTRY as critic_resigtry
import torch as th
from torchsummary import summary


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.scheme = scheme

        if self.args.separated_policy:
            self._build_comm(input_shape)
            self._build_agents(self.args.msg_out_size)
            # build the critic
            self._build_critic(self.args.msg_out_size)
        else:
            self._build_agents(input_shape)

        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        # print(ep_batch['obs'].shape)
        
        # Build the communication outputs from the gnn, which are inputs to both actor and critic
        if self.args.separated_policy:
            comm_input = self._build_inputs(ep_batch, t)
            agent_inputs = self._build_msg(comm_input, ep_batch.batch_size, ep_batch["adj_matrix"][:, t, ...], ep_batch.device)# self._build_inputs(ep_batch, t)
        else:
            agent_inputs = self._build_inputs(ep_batch, t)

        
        avail_actions = ep_batch["avail_actions"][:, t]

        # use this if just the actor is a gnn.
        if self.args.agent == "gnn" or self.args.agent == "gat" or self.args.agent == "dual_channel_gnn" or self.args.agent == "dual_channel_gat":
            agent_outs, _ = self.agent(agent_inputs, ep_batch["adj_matrix"][:, t, ...])
        
        # use gnn has the gnn as the comm layer, and mlps for both actor and critic. (like GPPO)
        elif self.args.use_gnn:
            agent_outs, _ = self.agent(agent_inputs)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if self.args.use_gnn:
                if not test_mode:
                    epsilon_action_num = agent_outs[-1]

                    if getattr(self.args, "mask_before_softmax", True):
                        # select random action with probability epsilon
                        epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).floor()

                    agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                                        + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)
            
                    if getattr(self.args, "mask_before_softmax", True):
                        # Zero out the unavailable actions
                        agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        if self.args.use_gnn:
            self.gnn.cuda_transfer()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.agent, "{}/agent_model.pt".format(path))
        if self.args.separated_policy:
            th.save(self.gnn.state_dict(), "{}/gnn.th".format(path))
            th.save(self.agent, "{}/gnn_model.pt".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.separated_policy:
            self.gnn.load_state_dict(th.load("{}/gnn.th".format(path), map_location=lambda storage, loc: storage))
  
    def _build_agents(self, input_shape):
        """
        Agent is the actor portion of the policy
        """
        print("\033[31m" + self.args.agent + "\033[0m")
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        # summary(self.agent, input_size=input_shape)
        print("\033[31m" + str(type(self.agent)) + "\033[0m")

    def _build_critic(self, input_shape):
        """
        Critic network
        """
        print("\033[31m" + self.args.critic_type + "\033[0m")
        self.critic = critic_resigtry[self.args.critic_type](self.scheme, self.args)

    def _build_comm(self, input_shape):
        self.gnn = comm_REGISTRY["gcn"](input_shape, self.args)
        # self.gnn = agent_REGISTRY["gnn"](input_shape, self.args)

    def _build_msg(self, batch, batch_size, adj_matrix, device):
        """
        If the gnn is used to feed to both the actor and critic, it goes here
        """
        input_observation = batch.reshape(batch_size, self.n_agents, -1).to(device=device)
        msg_enc = self.gnn(input_observation, th.tensor(adj_matrix, device=device))
        reshaped_msg_enc = msg_enc.reshape(batch_size * self.n_agents, -1)
        return reshaped_msg_enc

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
