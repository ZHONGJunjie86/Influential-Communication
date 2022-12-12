import torch.multiprocessing as mp

class Shared_Data(): #nn.Module
    def __init__(self, args, agent_name_list):
        # super(Shared_Data, self).__init__()
        # sys
        self.shared_lock = mp.Manager().Lock()
        self.event = mp.Event()
        # self.shared_count = mp.Value("d", 0)
        self.process_num = args.processes
        # loss
        self.a_loss_agent = mp.Value("d", 0.0)
        self.c_loss_agent = mp.Value("d", 0.0)
        self.a_loss_adversary = mp.Value("d", 0.0)
        self.c_loss_adversary = mp.Value("d", 0.0)
        # reeard
        # "adversaries all reward"; "agents all reward"; ""relative_reward""
        self.adversaries_all_reward = mp.Manager().list([])
        self.agents_all_reward = mp.Manager().list([])
        self.relative_reward = mp.Manager().list([])
    
    def send(self, loss_dict_agent, loss_dict_adversary, send_process_data_dict):
        self._reset()
        self.a_loss_agent.value = loss_dict_agent['a_loss']
        self.c_loss_agent.value = loss_dict_agent['c_loss']
        self.a_loss_adversary.value = loss_dict_adversary['a_loss']
        self.c_loss_adversary.value = loss_dict_adversary['c_loss']

        for i in range(self.process_num):
            self.adversaries_all_reward.append(send_process_data_dict[i]["adversaries all reward"])
            self.agents_all_reward.append(send_process_data_dict[i]["agents all reward"])
            self.relative_reward.append(send_process_data_dict[i]["relative_reward"])
        
        self.event.set()
        self.event.clear()
        
    def _reset(self):
        del self.adversaries_all_reward[:]
        del self.agents_all_reward[:]
        del self.relative_reward[:]
        self.a_loss_agent.value = 0.0
        self.c_loss_agent.value = 0.0
        self.a_loss_adversary.value = 0.0
        self.c_loss_adversary.value = 0.0