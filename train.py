from train_env.step import step
from system.utils import Shared_Data
from config.config import *
import torch.multiprocessing as mp

if __name__ == "__main__":
    # loggers
    shared_data = Shared_Data(args, agent_name_list)
    processes = []
    for rank in range(args.processes+1):
        p = mp.Process(target=step, args=(rank, shared_data))
        p.start()
        processes.append(p)
    for p in processes: 
        p.join()
        