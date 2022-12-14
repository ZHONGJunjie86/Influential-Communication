import math
import torch
import torch.nn as nn
min_dis = 0.8 # max_all_adv_reward = 0.15
max_reward = 0.02

kl_loss = nn.KLDivLoss(reduction="batchmean")

def compute_dis(my_pos, other_pos):
    return other_pos[0]**2 + other_pos[1]**2# (my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2


def compute_dis_reward(distance_agent_dict, distance_reward_dict, agent_pos):
    for adv_name, pos in distance_agent_dict.items():
        dis = compute_dis(pos,  agent_pos)
        if dis < min_dis:
            distance_reward_dict[adv_name] = min(1/dis/1e5, max_reward)



def information_share(sta, rewards, agents, args, agent_name_list):
    # type reward
    type_reward_dict = {"adversary":0, "agent":0}
    for name in rewards.keys():
        if "adversary" in name:
            type_reward_dict["adversary"] += rewards[name]
        else:
            type_reward_dict["agent"] += rewards[name]
    
    
    # 只和最近的互换
    states = [list(i) for i in sta.values()]
    distance_reward_dict = {}
    distance_agent_dict = {}
    
    # adversary
    for i in range(args.num_adversaries):
        my_pos = states[i][2:4]
        distance_agent_dict[agent_name_list[i]] = my_pos
        distance_reward_dict[agent_name_list[i]] = 0
        min_adversary_index_position = [-1, float("inf")]
        other_adversary_index = 0

        start_point = 4 + args.num_obstacles * 2
        for j in range(args.num_adversaries): 
            if j == i:
                other_adversary_index += 1
                continue
            dis = compute_dis(my_pos, states[i][start_point: start_point+2])
            if dis < min_adversary_index_position[1]:
                min_adversary_index_position = [other_adversary_index, dis]
            
            start_point += 2
            other_adversary_index += 1
        
        # if min_adversary_index_position[0] != -1:
        #     agents.agents["adversary_" + str(i)].memory["follower"].follower_share_inform.append(
        #             agents.agents["adversary_" + str(min_adversary_index_position[0])].memory["leader"].hidden_states[-1])

    # agents
    for i in range(args.num_good):
        my_pos = states[i + args.num_adversaries][2:4]
        min_agent_index_position =  [-1, float("inf")]
        other_agent_index = 0
        compute_dis_reward(distance_agent_dict, distance_reward_dict, my_pos)

        
        start_point = 4 + args.num_obstacles * 2 + args.num_adversaries * 2
        for j in range(args.num_good): 
            if j == i:
                other_agent_index += 1
                continue
            dis = compute_dis(my_pos, states[i + args.num_adversaries][start_point: start_point+2])
            if dis < min_agent_index_position[1]:
                min_agent_index_position = [other_agent_index, dis]
            
            start_point += 2
            other_agent_index += 1
        
        # if min_agent_index_position[0] != -1:
        #     agents.agents["agent_" + str(i)].memory["follower"].follower_share_inform.append(
        #             agents.agents["agent_" + str(min_agent_index_position[0])].memory["leader"].hidden_states[-1])
        # else:
        #     agents.agents["agent_" + str(i)].memory["follower"].follower_share_inform.append(agents.agents["agent_" + str(i)].hidden_state_zero.numpy())
    
    return distance_reward_dict, type_reward_dict


def send_curve_data(total_step_reward, agent_type_list): 
    send_dic = {"relative_reward": sum(total_step_reward.values()),
                "agents all reward": 0, "adversaries all reward": 0}

    for name, reward in total_step_reward.items():
        if "adversar" in name:
            send_dic["adversaries all reward"] += reward
        else:
            send_dic["agents all reward"] += reward
    
    return send_dic


def collect_data(target, source, agent_type):
    target.old_states = torch.cat((target.old_states, source.old_states), dim = 0) 
    target.old_hidden_states = torch.cat((target.old_hidden_states,source.old_hidden_states), dim = 0) 
    target.old_actions = torch.cat((target.old_actions, source.old_actions), dim = 0) 
    target.old_logprobs = torch.cat((target.old_logprobs, source.old_logprobs), dim = 0) 
    target.old_values = torch.cat((target.old_values, source.old_values), dim = 0) 
    target.target_value = torch.cat((target.target_value, source.target_value), dim = 0) 
    target.GAE_advantage = torch.cat((target.GAE_advantage, source.GAE_advantage), dim = 0) 
        
    if agent_type != "agent":
        target.old_actions_com = torch.cat((target.old_actions_com, source.old_actions_com), dim = 0) 
        target.old_logprobs_com = torch.cat((target.old_logprobs_com, source.old_logprobs_com), dim = 0) 
        target.target_value_com = torch.cat((target.target_value_com, source.target_value_com), dim = 0) 
        target.GAE_advantage_com = torch.cat((target.GAE_advantage_com, source.GAE_advantage_com), dim = 0) 
        target.old_com = torch.cat((target.old_com, source.old_com), dim = 0) 


def compute_com_reward(agent_name_list, agent_kl_dict, beta, com_dim):
    com_reward_dict = {}
    for i, agent_name in enumerate(agent_name_list):
        if "agent" in agent_name:
            continue
        influence_reward = 0
        
        for other_name in agent_name_list:
            if agent_name == other_name or "agent" in other_name:
                continue
            
            other_real_logits = agent_kl_dict[other_name]["real_logits"].reshape(1, com_dim, 1)
            # p(j) * p(j|k)
            union_p = other_real_logits * agent_kl_dict[other_name][i].reshape(1, com_dim, com_dim)
            # sum_k p(j)
            sum_p = union_p.sum(dim=1,keepdim=True)
            # kl
            kl_reward = kl_loss(sum_p, agent_kl_dict[other_name]["real_logits"])
            influence_reward += beta * kl_reward
        com_reward_dict[agent_name] = influence_reward
    
    
    # if old_actions_com == None:
    #     for key, dis in kl_dict.items():
    #         kl_dict[key] = float(self.kl_loss(dis.reshape(batch_size, 1, self.action_dim_com), 
    #                                     logits.reshape(batch_size, 1, self.action_dim_com)).detach().cpu().numpy())
    
    return com_reward_dict
        