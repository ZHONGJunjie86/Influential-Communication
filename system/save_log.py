import wandb

def process_run(rank, shared_data):
    exp_name = "IC 22 fixV' a0.0 b1. per10 clip.2"
    wandb.init(
    project="Bi-Level-Actor-Critic-with-F", 
    entity="zhongjunjie",
    group=exp_name
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
        if rank == 1:
            print(exp_name)
        shared_data.event.wait()
        

    