import torch
from rl.policies import get_actor_critic_by_name
from rl.low_level_agent import LowLevelAgent
import gym
from rl.trainer import get_subdiv_space, get_agent_by_name, Trainer
from rl.config import argparser
from rl.main import make_log_files
from mpi4py import MPI
import numpy as np
from collections import OrderedDict
import signal
import os
from util.logger import logger
from util.pytorch import get_ckpt_path
import sys


def multi_agent_run(config):
    rank = MPI.COMM_WORLD.Get_rank()
    config.rank = rank
    config.is_chef = rank == 0
    config.seed = config.seed + rank
    config.num_workers = MPI.COMM_WORLD.Get_size()


    if config.is_chef:
        logger.warn('Run a base worker')
        """
        Sets up log directories and saves git diff and command line.
        """
        config.run_name = 'rl.{}.{}.{}'.format(config.env, config.prefix, config.seed)

        config.log_dir = os.path.join(config.log_root_dir, config.run_name)
        logger.info('Create log directory: %s', config.log_dir)
        os.makedirs(config.log_dir, exist_ok=True)

        if config.is_train:
            config.record_dir = os.path.join(config.log_dir, 'video')
        else:
            config.record_dir = os.path.join(config.log_dir, 'eval_video')
        logger.info('Create video directory: %s', config.record_dir)
        os.makedirs(config.record_dir, exist_ok=True)

        if config.subdiv_skill_dir is None:
            config.subdiv_skill_dir = config.log_root_dir



    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
 
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # set global seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.virtual_display is not None:
        os.environ["DISPLAY"] = config.virtual_display

    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
        assert torch.cuda.is_available()
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")

    # trainer contains all we need
    trainer = Trainer(config)

    ckpt_path, ckpt_num = get_ckpt_path(config.log_dir, None)

    if ckpt_path is not None:
        logger.warn('Load checkpoint %s', ckpt_path)
        ckpt = torch.load(ckpt_path)
        trainer._meta_agent.load_state_dict(ckpt['meta_agent'])
        trainer._agent.load_state_dict(ckpt['agent'])

    env = trainer._env

    ob = env.reset()
    print("ob:", ob)

    meta_ac, meta_ac_before_activation, meta_log_prob = trainer._meta_agent.act(ob, is_train=False)

    print("meta_ac:", meta_ac)
    print("meta_ac_before_activation:", meta_ac_before_activation)

    ll_ob = ob.copy() 
    ac, ac_before_activation = trainer._agent.act(ll_ob, meta_ac, is_train=False)
    print("ac:", ac)
    print("ac_before_activaion:", ac_before_activation)
    input()

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


    print(ob_space)
    print(clusters)    
    
    for cluster in clusters:
        ob_space[','.join(cluster[0]) + '_diayn'] = config.z_dim
    
    print(ob_space)

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
    multi_agent_run(args)
