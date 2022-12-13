import wandb

def process_run(rank, shared_data):
    wandb.init(
    project="Bi-Level-Actor-Critic-with-F", 
    entity="zhongjunjie",
    group="IC 3 14p"
    )
    wandb.config = {
    "learning_rate": 0.0003,
    }  
    
    shared_data.event.wait()
    while True:
        send_dict = {}
        
        shared_data.shared_lock.acquire()
        if rank == 1:
            send_dict['a_loss_agent'] = shared_data.a_loss_agent.value
            send_dict['a_loss_agent'] = shared_data.c_loss_agent.value
            send_dict['a_loss_adversary'] = shared_data.a_loss_adversary.value
            send_dict['a_loss_adversary'] = shared_data.c_loss_adversary.value
        
        send_dict["adversaries all reward"] = shared_data.adversaries_all_reward[rank-1]
        send_dict["agents all reward"] = shared_data.agents_all_reward[rank-1]
        send_dict["relative_reward"] = shared_data.relative_reward[rank-1]
        shared_data.shared_lock.release()

        wandb.log(send_dict)
        # print(send_dict)
        shared_data.event.wait()
        

    