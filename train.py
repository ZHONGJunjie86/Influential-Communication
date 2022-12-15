from train_env.step import step
from system.utils import Shared_Data
from config.config import *
import sys
import torch.multiprocessing as mp

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        # or else you get a deadlock in conv2d
        raise "Must be using Python 3 with linux!"
    # loggers
    shared_data = Shared_Data(args, agent_name_list)
    processes = []
    for rank in range(args.processes+1):
        p = mp.Process(target=step, args=(rank, shared_data))
        p.start()
        processes.append(p)
    for p in processes: 
        p.join()
        