from train_env.utils import information_share, send_curve_data, collect_data, compute_com_reward
from config.config import *
from algorithm.PPO import PPO
from algorithm.network import ActorCritic
import time
from system.utils import Shared_Data
from system.save_log import process_run 
import torch.multiprocessing as mp
import collections

def step(rank, shared_data):
    if rank !=0:
        process_run(rank, shared_data)
    # initialize
    agents_net = {"agent": ActorCritic("agent", obs_shape_by_type["agent"], 5, device).to(device), 
                  "adversary":ActorCritic("adversary", obs_shape_by_type["adversary"], 5, device, args.com_dim, args.num_adversaries).to(device)}
    agents_optimizer = {name: torch.optim.Adam(agents_net[name].parameters(),
                                               lr=args.a_lr) for name in agents_net}
    agents_process_dict = {}
    for i in range(args.processes):
        agents_process_dict[i] = {name : PPO(name, args, obs_shape_by_type, device) for name in agent_name_list} 
                                
    send_process_data_dict = {i:{"a_loss":0.0, "c_loss":0.0} for i in range(args.processes)}
    RENDER = False # True
    episode = 0    
    lr = args.a_lr

    while episode < args.max_episodes:
        print("-----------------Episode: ", episode)
            
        for i in range(args.processes):
            agents = agents_process_dict[i]
            env = simple_tag_v2.parallel_env(num_good=args.num_good, num_adversaries=args.num_adversaries,
                                 num_obstacles=args.num_obstacles, max_cycles=args.max_cycles, continuous_actions=False)
            env.seed(seed = None)

            # reset
            for agent_name in agents:
                agents[agent_name].reset()
            total_step_reward = collections.defaultdict(float)
            states = env.reset()
            dones = {}
            rewards = {}
            agent_com_dict = {}
            agent_kl_dict = {}
            com_reward_dict = {}
            for name in agent_name_list:
                if "agent" in name:
                    continue
                agent_com_dict[name] = -1
                agent_kl_dict[name] = {} 
                com_reward_dict[name] = 0
            step = 0
            if RENDER and episode % 10 == 0:
                env.render()
                time.sleep(0.1)

            for name in agent_name_list:
                dones[name] = False
                rewards[name] = 0

            while True:
                ################################# collect  action #############################
                actions = {}
                com = [agent_com_dict[i] for i in agent_com_dict]
                
                # pass hidden states between agents who can see each other
                distance_reward_dict, type_reward_dict = information_share(states, rewards, agents, args, agent_name_list)
                
                for agent_name in agent_name_list:
                    if "agent" in agent_name:
                        reward = rewards[agent_name]/100
                        type_reward = type_reward_dict["agent"]
                        agent_type = "agent"
                        com_reward = 0
                    else: 
                        reward = rewards[agent_name]/10  + distance_reward_dict[agent_name]
                        # type_reward = type_reward_dict["adversary"]
                        agent_type = "adversary"
                        com_reward = com_reward_dict[agent_name] + args.alpha * reward
                    
                    total_step_reward[agent_name] += reward

                    action, action_com, kl_dict = agents[agent_name].choose_action(states[agent_name], com,
                                                reward, float(com_reward), dones[agent_name], agents_net[agent_type])
                    if "adversary" in agent_name:
                        agent_kl_dict[agent_name] = kl_dict 
                        agent_com_dict[agent_name] = action_com
                    actions[agent_name] = action

                com_reward_dict = compute_com_reward(agent_name_list, agent_kl_dict, args.beta, args.com_dim)
                states, rewards, dones, infos = env.step(actions)
                step += 1

                if RENDER and episode % 10 == 0:
                    env.render()
                    time.sleep(0.1)
                    
                ################################# env rollout ##########################################
                # ================================== collect data & update ========================================
                if True in dones.values():
                    # last_reward   type_reward_dict
                    for agent_name in agent_name_list:
                        if "agent" in agent_name:
                            reward = rewards[agent_name]/100
                            agents[agent_name].last_reward(reward, None, dones[agent_name])
                        else: 
                            reward = rewards[agent_name]/10  + distance_reward_dict[agent_name]
                            agents[agent_name].last_reward(reward, com_reward_dict[agent_name], dones[agent_name])
                        total_step_reward[agent_name] += reward
                
                    break
            
            # compute GAE
            for agent_name in agent_name_list:
                agents[agent_name].compute_GAE()

            send_process_data_dict[i] = send_curve_data(total_step_reward, agent_type_list)
            
        # collect data
        agent_target = agents_process_dict[0]["agent_0"]
        adversary_target = agents_process_dict[0]["adversary_0"]
        for i in range(args.processes):
            agents = agents_process_dict[i]
            for agent_name in agent_name_list:
                if i == 0 and (agent_name == "adversary_0" or agent_name == "agent_0"):
                    continue
                if "agent" in agent_name:
                    collect_data(agent_target, agents[agent_name], "agent")
                else: 
                    collect_data(adversary_target, agents[agent_name], "adversary")
        
        # train 
        loss_dict_agent = agent_target.train_with_shared_data(agents_net["agent"], agents_optimizer["agent"])
        loss_dict_adversary = adversary_target.train_with_shared_data(agents_net["adversary"], agents_optimizer["adversary"])
        shared_data.send(loss_dict_agent, loss_dict_adversary, send_process_data_dict)

        print("Episode ", episode, " over. lr: ", lr)
        if episode % 15 == 0:
            agent_target.save_model(model_save_path, agents_net["agent"])
            adversary_target.save_model(model_save_path, agents_net["adversary"])
            if episode!= 0 and episode % 10 == 0 and lr>args.min_lr:
                lr = max(lr  * args.lr_decay, args.min_lr)
                agents_optimizer = {name: torch.optim.Adam(agents_net[name].parameters(),
                                               lr=lr) for name in agents_net}
                
        episode += 1





