import torch
from rl.policies import get_actor_critic_by_name
from rl.low_level_agent import LowLevelAgent
import gym
from rl.trainer import get_subdiv_space, get_agent_by_name
from rl.config import argparser
from mpi4py import MPI
import numpy as np
from collections import OrderedDict


def basic_single_agent_run(config):

    rank = MPI.COMM_WORLD.Get_rank()
    config.rank = rank
    config.is_chef = rank == 0
    config.seed = config.seed + rank
    config.num_workers = MPI.COMM_WORLD.Get_size()


    ''' environment '''
    env_config = {'husky': 1.0, 'skill': 'push'}
    env = gym.make('husky-forward-v0', **env_config)

    ob_space, ac_space, clusters = get_subdiv_space(env, None)

    for cluster in clusters:
        ob_space[','.join(cluster[0]) + '_diayn'] = config.z_dim

    actor, critic = get_actor_critic_by_name("mlp")
    
    config.device = torch.device("cuda")

    agent = get_agent_by_name('sac')(config, ob_space, ac_space, actor, critic)
     
    ckpt_file_addr = '/home/hcrlab01/LiChen_workspace/transformer_robots_coordination/coordination/log/rl.husky-forward-v0.h-1-push-dim-4.1/ckpt_03990076.pt'
    ckpt = torch.load(ckpt_file_addr) 
    

    agent.load_state_dict(ckpt['agent'])
    

    obs = OrderedDict([
        ("husky_1", np.random.randn(30)),
        ("box_1", np.random.randn(9)),
        ("husky_1,box_1_diayn", np.random.randn(4))
    ])
    

    ll_ob = obs.copy()
    ac, ac_t = agent.act(ll_ob, is_train=False)

    print("---------- Basic Single Agent ----------")
    print(ac)
    print(ac_t)
    print("basic_single_agent_run done")

if __name__ == '__main__':
    args, unparsed = argparser()
    #basic_single_agent_run(args)
