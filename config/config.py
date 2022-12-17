import pettingzoo.mpe.simple_tag_v2 as simple_tag_v2
import argparse
import datetime
import sys
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', default=0.95, type=float)  # 0.1**(1/50)
parser.add_argument('--a_lr', default= 1.5e-3, type=float)  # 0.0015
parser.add_argument('--min_lr', default= 1.2e-5, type=float)  # 0.00002  1.2e-5
parser.add_argument('--lr_decay', default= 0.95, type=float)


parser.add_argument('--render', action='store_true')
parser.add_argument("--save_interval", default=20, type=int)  # 1000
parser.add_argument("--model_episode", default=0, type=int)
parser.add_argument(
    '--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
# env
parser.add_argument("--num_good",  default=1, type=int)  
parser.add_argument("--num_adversaries",  default=3, type=int)  
parser.add_argument("--num_obstacles",  default=2, type=int) 
parser.add_argument("--max_cycles", default=150, type=int)  # Agent Environment Cycle 等于游戏步
parser.add_argument("--max_episodes", default=10000000, type=int)

# experiment
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--beta", default=1.0, type=float)
parser.add_argument("--com_dim", default=9, type=int)


# PPO
parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
parser.add_argument("--load_model_run", default=8, type=int)
parser.add_argument("--load_model_run_episode", default=4000, type=int)
parser.add_argument("--K_epochs", default=4, type=int)
parser.add_argument("--clip", default=0.2, type=float)

# Multiprocessing
parser.add_argument('--processes', default=14, type=int,
                    help='number of processes to train with')
                                
args = parser.parse_args()


# 环境相关
env = simple_tag_v2.parallel_env(num_good=args.num_good, num_adversaries=args.num_adversaries,
                                 num_obstacles=args.num_obstacles, max_cycles=args.max_cycles, continuous_actions=False)
agent_name_list = [agent_name for agent_name in env.possible_agents]


agent_type_list = ["agent", "adversary"]
obs_shape_by_type = {"agent": 4 + 2 * args.num_obstacles + 2* (args.num_good + args.num_adversaries - 1) + 2 * (args.num_good - 1), 
                     "adversary": 4 + 2 * args.num_obstacles + 2 * (args.num_good + args.num_adversaries - 1) + 2 * args.num_good}


# 定义保存路径
path = "/home/j-zhong/work_place/Influential-Communication/model/model1/"
model_load_path = {"agent": path, 
                   "adversary":path}
model_save_path = {"agent": path, 
                   "adversary":path}

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # 




    