from unicodedata import name
from numpy import tri
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import copy
# torch.set_default_tensor_type(torch.DoubleTensor)


class ActorCritic(nn.Module):        
    def __init__(self, agent_type, obs_size, action_dim, device, action_dim_com = None, com_dim = None):  #(n+2p-f)/s + 1 
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_dim_com = action_dim_com
        self.obs_size = obs_size
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_size = 64
        self.com_dim = com_dim
        self.agent_type = agent_type
        self.device = device
        if self.agent_type != "agent":
            self.all_action_com = torch.Tensor([i for i in range(self.action_dim_com)]
                                 ).reshape(1,self.action_dim_com,1).to(self.device)


        # 
        self.linear1 = nn.Linear(obs_size, 64)
        self.self_attention = nn.MultiheadAttention(64, 4)        
        self.gru = nn.GRU(input_size = 64, hidden_size = 64, num_layers = 1, batch_first=True)
        
        # actor
        self.linear_actor = nn.Linear(64, action_dim)
        self.categorical_dis = torch.distributions.Categorical

        # critic        
        self.linear_critic = nn.Linear(64, action_dim)

        # communicate part
        if agent_type == "adversary":
            self.linear_com = nn.Linear(self.com_dim, 30)
            self.linear_2 = nn.Linear(64 + 30, 64)

            self.categorical_dis_com = torch.distributions.Categorical
            self.kl_loss = nn.KLDivLoss(reduction="batchmean")
            self.linear_actor_com = nn.Linear(64, action_dim_com)
            self.linear_critic_com = nn.Linear(64, action_dim_com)
 
        self.initialize()

    
    def initialize(self):
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        torch.nn.init.kaiming_normal_(self.linear_actor.weight)
        torch.nn.init.kaiming_normal_(self.linear_critic.weight)

        if self.agent_type == "adversary":
            torch.nn.init.kaiming_normal_(self.linear_com.weight)
            torch.nn.init.kaiming_normal_(self.linear_2.weight)
            torch.nn.init.kaiming_normal_(self.linear_actor_com.weight)
            torch.nn.init.kaiming_normal_(self.linear_critic_com.weight)
        

    def forward(self, obs, obs_com, h_old, old_actions = None, old_actions_com = None): 
        batch_size = obs.size()[0]
            
        x = F.relu(self.linear1(obs)) # .float()
        x = x.view(batch_size, 1, -1)
        x_before_rnn = self.self_attention(x,x,x)[0] + x

        if self.agent_type == "adversary":
            # real
            if old_actions_com == None:
                com = torch.Tensor(obs_com).to(self.device)
            else:
                com = obs_com
            com = com.view(batch_size, 1, self.com_dim)
            x_com = F.relu(self.linear_com(com/10))
            x = torch.cat([x_before_rnn.view(batch_size, 1, -1), 
                           x_com.reshape(batch_size, 1, 30)    
                            ], -1).view(batch_size, 1, -1)
            x = F.relu(self.linear_2(x))
            x,h_state = self.gru(x, h_old.detach())
        else:
            x,h_state = self.gru(x_before_rnn, h_old.detach())
        
        # actor
        logits = self.softmax(self.linear_actor(x))
        dis =  self.categorical_dis(logits.reshape(batch_size, 1, self.action_dim))
        
        if old_actions != None:
            action = old_actions.view(batch_size,-1)
        else:
            action = dis.sample()
        entropy = dis.entropy().mean()
        selected_log_prob = dis.log_prob(action)

        # critic
        value = self.linear_critic(x)
        action_value = value.gather(-1, action.unsqueeze(1).long())
        value = value.mean(-1, keepdim = True)


        # communicate part 
        if self.agent_type == "adversary":

            kl_dict = {}
            # com part
            logits = self.softmax(self.linear_actor_com(x))
            dis =  self.categorical_dis_com(logits.reshape(batch_size, 1, self.action_dim_com))
            if old_actions_com != None:
                action_com = old_actions_com.view(batch_size,-1)
            else:
                action_com = dis.sample()
            entropy_com = dis.entropy().mean()
            selected_log_prob_com = dis.log_prob(action)

            # critic
            value_com = self.linear_critic_com(x)
            action_value_com = value_com.gather(-1, action_com.unsqueeze(1).long())
            value_com = value_com.mean(-1, keepdim = True)

            kl_dict["real_logits"] = logits            


            # contrafactual
            if old_actions_com == None:
                # obs_com: list
                for i in range(len(obs_com)):
                    com = copy.deepcopy(obs_com)
                    com = torch.Tensor(com).reshape(1,1,self.com_dim).to(self.device)
                    com = com.repeat(1,self.action_dim_com,1)
                    x_com = torch.cat([com[:,:,:i], self.all_action_com, com[:,:,i+1:]],2)
                    x_com = F.relu(self.linear_com(x_com/10))
                    x = torch.cat([x_before_rnn.view(batch_size, 1, -1).repeat(1,self.action_dim_com,1), 
                                      x_com.reshape(batch_size, self.action_dim_com, 30)   
                                      ], -1).view(batch_size, self.action_dim_com, -1)
                    x = F.relu(self.linear_2(x))
                    x,_ = self.gru(x, h_old.detach())

                    logits = self.softmax(self.linear_actor_com(x))
                    kl_dict[i] = logits

            return value.reshape(batch_size,1,1), action,selected_log_prob, value_com.reshape(batch_size,1,1), action_com,\
                selected_log_prob_com , h_state.detach().data, entropy, entropy_com, kl_dict, action_value.reshape(batch_size,1,1), action_value_com.reshape(batch_size,1,1)
        else:
            return value.reshape(batch_size,1,1), action,selected_log_prob, None, None,\
                None, h_state.detach().data, entropy, None, None, action_value.reshape(batch_size,1,1), None

        # value, action, logprobs, value_com, action_com,\
            # logp_com, h_s, entropy, entropy_com, _