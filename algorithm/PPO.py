import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
import torch.nn as nn
from torch.autograd import Variable
from algorithm.utils import Memory
import torch.nn.functional as F
import copy


class PPO():
    def __init__(self, agent_name, args, obs_shape_by_type, device):
        self.device = device
        self.a_lr = args.a_lr
        self.gamma = args.gamma
        self.adversary_num = args.num_adversaries
        if "agent" in agent_name:
            self.agent_type = "agent"
            self.action_dim = {"action":5}
        else:
            self.agent_type = "adversary"
            self.action_dim = {"action":5, "communicate":9}

        self.use_upgo = True
        self.use_gae = False
        #
        self.obs_shape = obs_shape_by_type[self.agent_type]
        self.com_shape = self.adversary_num

        # train
        self.eps_clip = 0.2
        self.vf_clip_param = 0.2
        self.lam = 0.95
        self.K_epochs = args.K_epochs
        self.old_value_1, self.old_value_2 = 0, 0
        self.entropy_coef = 0.01
        self.hidden_size = 64
        self.hidden_state_zero = torch.zeros(1,1,self.hidden_size).to(self.device).detach()
        self.com_zero = torch.zeros(1,1,self.com_shape)

        # social reward coef
        self.env_coef = 0.5
        self.social_coef = 0.5
        self.reward_follower_last = 0

        # for K updates
        self.advantages = {}
        self.target_value = {}
        

        self.SmoothL1Loss = torch.nn.SmoothL1Loss()

        #
        self.memory = Memory()
        self.loss_name_list = ["a_loss", "c_loss", "entropy"]
        self.loss_dic = {}
        for loss_name in self.loss_name_list:
            self.loss_dic[loss_name] = 0

        #
        self.reset()


    def choose_action(self, state, com, reward, com_reward, done, actor):
        
        obs_tensor = torch.Tensor(state).to(self.device).reshape(1,self.obs_shape)    
        reward = torch.Tensor([reward]).to(self.device).reshape(1,1)     
        done = torch.Tensor([done]).to(self.device).reshape(1,1)    

        if self.memory.hidden_states == []:
            self.memory.hidden_states.append(self.hidden_state_zero)
        h_s = self.memory.hidden_states[-1]

        if self.agent_type != "agent":
            com_tensor = com   
        else:
            com_tensor = None 

        value, action, logp, value_com,\
            action_com, logp_com, h_s, _, _, dis_dict,\
                action_value, action_value_com = actor(obs_tensor, com_tensor, h_s)

        with torch.no_grad():
            self.memory.states.append(obs_tensor)
            self.memory.is_terminals.append(done)
            self.memory.actions.append(action)
            self.memory.logprobs.append(logp) 
            self.memory.hidden_states.append(h_s)
            self.memory.values.append(value)
            self.memory.action_values.append(action_value)
            self.memory.rewards.append(reward)

            if self.agent_type != "agent": # TODO ?
                com_reward = torch.Tensor([com_reward]).to(self.device).reshape(1,1)     
                self.memory.rewards_com.append(com_reward)
                self.memory.values_com.append(value_com)
                self.memory.actions_com.append(action_com)
                self.memory.logprobs_com.append(logp_com)
                self.memory.action_values_com.append(action_value_com)
                com_tensor = torch.Tensor(com_tensor).to(self.device).reshape(1,self.com_shape) 
                self.memory.com.append(com_tensor)

        if self.agent_type == "agent":
            return int(action.detach().cpu().numpy()), None, None
        else:
            return int(action.detach().cpu().numpy()), int(action_com.detach().cpu().numpy()), dis_dict


    def compute_GAE(self, ):
                    
        # Monte Carlo estimate of rewards:
        rewards = []
        GAE_advantage = [] 
        target_value = []  
        #
        discounted_reward = self.memory.values[-1] 
        action_value_pre = self.memory.action_values[-1]
        value_pre = self.memory.values[-1]
        advatage = action_value_pre - value_pre
        adv_gae = action_value_pre - value_pre
        g_t_pre = action_value_pre if action_value_pre >= value_pre  \
                                    else value_pre
                                    
        
        for reward, is_terminal, value, action_value in zip(reversed(self.memory.rewards[1:-1]), reversed(self.memory.is_terminals[:-1]),
                                                reversed(self.memory.values[:-1]), reversed(self.memory.action_values[:-1])): #反转迭代
            
            # reward = reward
            # is_terminal = is_terminal

            discounted_reward = reward + self.gamma *discounted_reward
            rewards.append(discounted_reward) #插入列表

            delta = reward + self.gamma*action_value_pre - value   # (1-is_terminal)*
            
            adv_gae = delta + self.gamma*self.lam*adv_gae 
            
            if action_value_pre >= value_pre:
                g_t = reward + self.gamma*g_t_pre
            else:
                g_t = reward + self.gamma*value_pre
            adv_upgo = g_t - value
            g_t_pre = g_t

            
            if self.use_gae:
                advatage = adv_gae 
            elif self.use_upgo:
                advatage = adv_upgo
            else:
                advatage = delta

            GAE_advantage.append(advatage) #插入列表
            target_value.append(float(value) + advatage)#)
            action_value_pre = action_value
            # value_pre = value # !
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards[::-1]).to(self.device).view(-1,1,1)
        self.target_value = torch.tensor(target_value[::-1]).to(self.device).view(-1,1,1)
        self.GAE_advantage = torch.tensor(GAE_advantage[::-1]).to(self.device).view(-1,1,1)
        # self.advantages = GAE_advantage
        
        if self.agent_type != "agent":
            self._compute_com_GAE()

        # shrink
        self.old_states = torch.stack(self.memory.states[:-1], dim=0) 
        self.old_hidden_states = torch.stack(self.memory.hidden_states[:-2], dim=0)
        self.old_actions = torch.stack(self.memory.actions[:-1], dim=0)
        self.old_logprobs = torch.stack(self.memory.logprobs[:-1], dim=0)
        self.old_values = torch.stack(self.memory.values[:-1], dim=0)
        self.old_action_values = torch.stack(self.memory.action_values[:-1], dim=0)

        if self.agent_type != "agent":
            self.old_actions_com = torch.stack(self.memory.actions_com[:-1], dim=0)
            self.old_logprobs_com = torch.stack(self.memory.logprobs_com[:-1], dim=0)
            self.old_com = torch.stack(self.memory.com[:-1], dim=0)
            self.old_values_com = torch.stack(self.memory.values_com[:-1], dim=0)
            self.old_action_values_com = torch.stack(self.memory.action_values_com[:-1], dim=0)

        return None

    def _compute_com_GAE(self):
        # Monte Carlo estimate of rewards:
        rewards = []
        GAE_advantage = [] 
        target_value = []  
        #
        discounted_reward = self.memory.values_com[-1] 
        action_value_pre = self.memory.action_values_com[-1]
        value_pre = self.memory.values_com[-1]
        advatage = action_value_pre - value_pre
        adv_gae = action_value_pre - value_pre
        g_t_pre = action_value_pre if action_value_pre >= value_pre  \
                                    else value_pre
                                    
        
        for reward, is_terminal, value, action_value in zip(reversed(self.memory.rewards_com[1:-1]), reversed(self.memory.is_terminals[:-1]),
                                                reversed(self.memory.values_com[:-1]), reversed(self.memory.action_values_com[:-1])): #反转迭代
            
            # reward = reward
            # is_terminal = is_terminal

            discounted_reward = reward + self.gamma *discounted_reward
            rewards.append(discounted_reward) #插入列表

            delta = reward + self.gamma*action_value_pre - value   # (1-is_terminal)*
            
            adv_gae = delta + self.gamma*self.lam*adv_gae 
            
            if action_value_pre >= value_pre:
                g_t = reward + self.gamma*g_t_pre
            else:
                g_t = reward + self.gamma*value_pre
            adv_upgo = g_t - value
            g_t_pre = g_t

            
            if self.use_gae:
                advatage = adv_gae 
            elif self.use_upgo:
                advatage = adv_upgo
            else:
                advatage = delta

            GAE_advantage.append( advatage) #插入列表
            target_value.append(float(value) + advatage)#)
            action_value_pre = action_value
            # value_pre = value # !
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards[::-1]).to(self.device).view(-1,1,1)
        self.target_value_com = torch.tensor(target_value[::-1]).to(self.device).view(-1,1,1)
        self.GAE_advantage_com = torch.tensor(GAE_advantage[::-1]).to(self.device).view(-1,1,1)
        # self.advantages = GAE_advantage


    def train_with_shared_data(self, actor, actor_optmizer):
        with torch.no_grad():
            self.old_states = self.old_states.view(-1,1,self.obs_shape)
            self.old_hidden_states = self.old_hidden_states.view(1, -1, self.hidden_size) # 
            self.old_actions = self.old_actions.view(-1,1,1)
            self.old_logprobs = self.old_logprobs.view(-1,1,1)
            self.old_values = self.old_values.view(-1,1,1)
            self.old_action_values = self.old_action_values.view(-1,1,1)
            self.target_value = self.target_value.view(-1,1,1)
            self.GAE_advantage = self.GAE_advantage.view(-1,1,1)
            self.GAE_advantage = (self.GAE_advantage - self.GAE_advantage.mean()) / (self.GAE_advantage.std() + 1e-6) 
                
            if self.agent_type != "agent":
                self.old_actions_com = self.old_actions_com.view(-1,1,1)
                self.old_logprobs_com = self.old_logprobs_com.view(-1,1,1)
                self.old_com = self.old_com.view(-1,1,self.com_shape)
                self.target_value_com = self.target_value_com.view(-1,1,1)
                self.old_values_com = self.old_values_com.view(-1,1,1)
                self.old_action_values_com = self.old_action_values_com.view(-1,1,1)
                self.GAE_advantage_com = self.GAE_advantage_com.view(-1,1,1)
                self.GAE_advantage_com = (self.GAE_advantage_com - self.GAE_advantage_com.mean()) / (self.GAE_advantage_com.std() + 1e-6) 
                
            
        batch_size = self.old_states.size()[0]
        for _ in range(self.K_epochs):
            batch_sample = batch_size # int(batch_size / self.K_epochs) # 
            # indices = torch.randint(batch_size, size=(batch_sample,), requires_grad=False)
            with torch.no_grad():
                old_states = self.old_states#[indices]

                old_hidden = self.old_hidden_states#.reshape(-1,1,self.hidden_size)[indices].view(1, -1, self.hidden_size)
                old_logprobs = self.old_logprobs# [indices]
                advantages = self.GAE_advantage#[indices].detach()
                target_value = self.target_value#[indices]
                old_value = self.old_values
                old_action_value = self.old_action_values
                old_actions = self.old_actions

                if self.agent_type != "agent":
                    old_com = self.old_com
                    old_actions_com = self.old_actions_com
                    old_logprobs_com = self.old_logprobs_com
                    target_value_com = self.target_value_com
                    GAE_advantage_com = self.GAE_advantage_com
                    old_value_com = self.old_values_com
                    old_action_value_com = self.old_action_values_com
                else:
                    old_com = None
                    old_actions_com = None
                    old_logprobs_com = None
                    target_value_com = None
                    GAE_advantage_com = None

            
            value, action, logprobs, value_com, action_com,\
            logp_com, h_s, entropy, entropy_com, _, action_value, action_value_com = actor(old_states, old_com, old_hidden,
                                                                                    old_actions, old_actions_com)
    
            ratios = torch.exp(logprobs.view(batch_sample,1,-1) - old_logprobs.detach())

            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages 
            #Dual_Clip
            surr3 = torch.max(torch.min(surr1, surr2),3*advantages)
            #torch.min(surr1, surr2)#

            # value_pred_clip = old_value.detach() +\
            #     torch.clamp(value - old_value.detach(), -self.vf_clip_param, self.vf_clip_param)
            # critic_loss1 = (value - target_value.detach()).pow(2)
            # critic_loss2 = (value_pred_clip - target_value.detach()).pow(2)
            # critic_loss = 0.5 * torch.max(critic_loss1 , critic_loss2).mean()
            
            # critic_loss = torch.nn.SmoothL1Loss()(value, target_value)
            action_value_pred_clip = old_action_value.detach() +\
                torch.clamp(action_value - old_action_value.detach(), -self.vf_clip_param, self.vf_clip_param)
            critic_loss1 = (action_value - target_value.detach()).pow(2)
            critic_loss2 = (action_value_pred_clip - target_value.detach()).pow(2)
            critic_loss = 0.5 * torch.max(critic_loss1 , critic_loss2).mean()

            if self.agent_type != "agent":
                ratios_com = torch.exp(logp_com.view(batch_sample,1,-1) - old_logprobs_com.view(batch_sample,1,-1).detach())
                surr1_com = ratios_com * GAE_advantage_com
                surr2_com = torch.clamp(ratios_com, 1-self.eps_clip, 1+self.eps_clip)*GAE_advantage_com
                #Dual_Clip
                surr3_com = torch.max(torch.min(surr1_com, surr2_com),3*GAE_advantage_com)
                
                # c_loss_com = torch.nn.SmoothL1Loss()(target_value_com, value_com) 
                # value_pred_clip_com = old_value_com.detach() +\
                # torch.clamp(value_com - old_value_com.detach(), -self.vf_clip_param, self.vf_clip_param)
                # critic_loss1_com = (value_com - target_value_com.detach()).pow(2)
                # critic_loss2_com = (value_pred_clip_com - target_value_com.detach()).pow(2)
                # critic_loss_com = 0.5 * torch.max(critic_loss1_com, critic_loss2_com).mean()
                # critic_loss = critic_loss + critic_loss_com

                action_value_pred_clip_com = old_action_value_com.detach() +\
                                        torch.clamp(action_value_com - old_action_value_com.detach(), -self.vf_clip_param, self.vf_clip_param)
                critic_loss1_com = (action_value_com - target_value_com.detach()).pow(2)
                critic_loss2_com = (action_value_pred_clip_com - target_value_com.detach()).pow(2)
                critic_loss_com = 0.5 * torch.max(critic_loss1_com, critic_loss2_com).mean()
                critic_loss = critic_loss + critic_loss_com

                entropy = entropy + entropy_com
                actor_loss = -surr3.mean() - surr3_com.mean() - self.entropy_coef * entropy + 0.5 * critic_loss
            else:
                actor_loss = -surr3.mean() - self.entropy_coef * entropy + 0.5 * critic_loss
            
            # do the back-propagation...
            actor_optmizer.zero_grad()
            actor_loss.backward()
            actor_optmizer.step()

            self.loss_dic['a_loss'] += float(actor_loss.cpu().detach().numpy())
            self.loss_dic['c_loss'] += float(critic_loss.cpu().detach().numpy())
            self.loss_dic['entropy'] += float(entropy.cpu().detach().numpy())

        return self.loss_dic


    def last_reward(self, reward, com_reward, done):
        reward = torch.Tensor([reward]).to(self.device).reshape(1,1)   
        done = torch.Tensor([done]).to(self.device).reshape(1,1)   
        self.memory.is_terminals.append(done)
        self.memory.rewards.append(reward)
            
        if self.agent_type != "agent":
            com_reward = torch.Tensor([com_reward]).to(self.device).reshape(1,1) 
            self.memory.rewards_com.append(com_reward)

    def reset(self):
        
        self.memory.clear_memory() 
        
        for loss_name in self.loss_name_list:
            self.loss_dic[loss_name] = 0
        
        self.reward_follower_last = 0

    def load_model(self, model_load_path):

        # "path + agent/adversary + leader/follower + .pth"
        for name in self.agent_name_list:
            model_actor_path = model_load_path[self.agent_type]+ self.agent_type  + name + ".pth"
            #print(f'Actor path: {model_actor_path}')
            if  os.path.exists(model_actor_path):
                actor = torch.load(model_actor_path, map_location=self.device)
                self.actor[name].load_state_dict(actor)
                #print("Model loaded!")
            else:
                sys.exit(f'Model not founded!')

    def save_model(self, model_save_path, actor):
        # print("---------------save-------------------")
        # print("new_lr: ",self.a_lr)

        # "path + agent/adversary + leader/follower + .pth"
        model_actor_path = model_save_path[self.agent_type]+ self.agent_type + ".pth"
        torch.save(actor.state_dict(), model_actor_path)

    # TODO
    def quick_load_model(self, actor, new_model_dict):
        actor.load_state_dict(new_model_dict[self.agent_type].state_dict())

