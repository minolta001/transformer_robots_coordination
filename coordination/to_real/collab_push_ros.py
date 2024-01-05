#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from math import pow,atan2,sqrt,sin,cos
from tf.transformations import euler_from_quaternion
import numpy as np
import threading
import time
from rl.config import argparser

import torch
from rl.policies import get_actor_critic_by_name
from rl.low_level_agent import LowLevelAgent
import gym
from rl.trainer import get_subdiv_space, get_agent_by_name, Trainer
from rl.main import make_log_files
from mpi4py import MPI
import numpy as np
from collections import OrderedDict
import signal
import os
from util.logger import logger
from util.pytorch import get_ckpt_path
import sys

class collab_push():
    def __init__(self, args):
        rospy.init_node('husky_controller')

        # robot_1       Husky
        self.robot_1_velocity_publisher = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10) # publish Twist message to Husky

        self.robot_1_pose = Pose()
        self.robot_1_pose_subscriber = rospy.Subscriber('/mocap_node/Robot_1/Odom', Odometry, self.robot_1_pose_callback)   # receive husky pose data from tracking system 
        self.box_1_pose = Pose()
        self.box_1_pose_subscriber = rospy.Subscriber('/mocap_node/Box1/Odom', Odometry, self.box_1_pose_callback)

        # robot_2


        self.rate = rospy.Rate(20)

        #self.trainer = self.load_trainer(args)

        self.husky_push()


    def load_trainer(self, config):         # load meta agent and lower-level agent model
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

        return trainer
     

    # receive husky pose data from tracking system (posistion & orientation)        Husky
    def robot_1_pose_callback(self, data):
        self.robot_1_pose = data.pose.pose

    def box_1_pose_callback(self, data):
        self.box_1_pose = data.pose.pose


    def husky_push(self):

        time.sleep(5)

        vel_msg = Twist()
        robot_1_x = self.robot_1_pose.position.x 
        box_1_x = self.box_1_pose.position.x

        while not rospy.is_shutdown():
            #current_x = self.robot_1_pose.position.x
            box_1_x = self.box_1_pose.position.x
            print(box_1_x)


            if(box_1_x <= 0.5):
                for i in range(10):
                    vel_msg.linear.x = 0.2
                    vel_msg.angular.z = 0.0
                    self.robot_1_velocity_publisher.publish(vel_msg)

            elif(0.5 < box_1_x and box_1_x < 2):
                for i in range(10):
                    vel_msg.linear.x = 0.2
                    vel_msg.angular.z = -0.07
                    self.robot_1_velocity_publisher.publish(vel_msg)

            self.rate.sleep()
      
    
if __name__ == '__main__':
    args, unparsed = argparser()
    collab_push_team = collab_push(args) 