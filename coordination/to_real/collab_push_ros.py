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
import math


def husky_forward_kinematic(ac_L, ac_R):
        linear_vel = (ac_L + ac_R)
        angular_vel = (ac_R - ac_L) / 1.667     # the coefficient is tested on real robot
        return linear_vel, angular_vel


class collab_push():
    def __init__(self, args):
        rospy.init_node('husky_controller')

        self.last_time = None
        self.time_gap = 0





        # Husky
        # Bunker
        self.bunker_yaw_velocity = None
        self.bunker_linear_velocity = None
        self.bunker_last_pose = None
        self.bunker_cur_pose = Pose()
        self.bunker_pose_vel_subscriber = rospy.Subscriber('/mocap_node/Bunker/Odom', Odometry, self.bunker_pose_vel_callback)

        # get Box1 pose
        # get Box2 pose
        # get Box pose



        # robot_1       Husky
        '''
        self.robot_1_pose = Pose()
        self.robot_1_pose_subscriber = rospy.Subscriber('/mocap_node/Robot_1/Odom', Odometry, self.robot_1_pose_callback)   # receive husky pose data from tracking system 
        self.box_1_pose = Pose()
        self.box_1_pose_subscriber = rospy.Subscriber('/mocap_node/Box1/Odom', Odometry, self.box_1_pose_callback)
        '''
        #self.robot_1_velocity_publisher = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10) # publish Twist message to Husky

        # robot_2       Bunker
        #self.robot_2_velocity_publisher = rospy.Publisher('/smoother_cmd_vel', Twist, queue_size=10) # publish Twist message to Bunker


         

        self.rate = rospy.Rate(20)

        #self.trainer = self.load_trainer(args)
        self.vel_test()


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
            sys.exit(128+signal
)
    
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

    def bunker_pose_vel_callback(self, data):
        cur_time = data.header.stamp.secs + (data.header.stamp.nsecs / 1000000000)
        self.bunker_cur_pose = data.pose.pose


        if self.bunker_last_pose != None and self.last_time != None:
            delta_linear_x = self.bunker_cur_pose.position.x - self.bunker_last_pose.position.x
            delta_linear_y = self.bunker_cur_pose.position.y - self.bunker_last_pose.position.y

            bunker_cur_quat = np.array([self.bunker_cur_pose.orientation.w,
                                        self.bunker_cur_pose.orientation.x,
                                        self.bunker_cur_pose.orientation.y,
                                        self.bunker_cur_pose.orientation.z])

            bunker_last_quat = np.array([self.bunker_last_pose.orientation.w,
                                         self.bunker_last_pose.orientation.x,
                                         self.bunker_last_pose.orientation.y,
                                         self.bunker_last_pose.orientation.z])
            
            last_quat_conjugate = np.array([bunker_last_quat[0], -bunker_last_quat[1], -bunker_last_quat[2], -bunker_last_quat[3]])
            
            delta_quat = np.dot(bunker_cur_quat,last_quat_conjugate)
            print(delta_quat)
            angle = 2 * math.acos(delta_quat[0])

            time_gap = cur_time - self.last_time
            self.time_gap = time_gap

            if time_gap != 0:
                # linear velocity
                linear_x_velocity = delta_linear_x / self.time_gap
                linear_y_velocity = delta_linear_y / self.time_gap

                self.bunker_linear_velocity = [linear_x_velocity, linear_y_velocity, 0]

                # angular velocity
                self.bunker_yaw_velocity = angle / time_gap
                print("{0:.8f}".format(self.bunker_yaw_velocity))
                


        self.last_time = cur_time
        self.bunker_last_pose = self.bunker_cur_pose

    
        


    # receive husky pose data from tracking system (posistion & orientation)        Husky
    def robot_1_pose_callback(self, data):
        self.robot_1_pose = data.pose.pose 

    def box_1_pose_callback(self, data):
        self.box_1_pose = data.pose.pose

    def vel_test(self):
        while not rospy.is_shutdown():
            continue
    
    def bunker_push(self):
        time.sleep(5)
        vel_msg_1 = Twist()     # husky
        vel_msg_2 = Twist()     # bunker
        start_time = time.time()
        while not rospy.is_shutdown():
            print("husky/bunker working")
            if(time.time() - start_time < 4):
                vel_msg_1.linear.x = 0.0
                vel_msg_1.angular.z = 3
                vel_msg_2.linear.x = 0.0
                vel_msg_2.angular.z = 0.0
                self.robot_1_velocity_publisher.publish(vel_msg_1)
                self.robot_2_velocity_publisher.publish(vel_msg_2)
            elif(time.time() - start_time > 4 and time.time() - start_time < 30):
                vel_msg_1.linear.x = 0.0
                vel_msg_1.angular.z = 0.3
                vel_msg_2.linear.x = 0.0
                vel_msg_2.angular.z = -0.3
                self.robot_1_velocity_publisher.publish(vel_msg_1)
                self.robot_2_velocity_publisher.publish(vel_msg_2)
            else:
                vel_msg_1.linear.x = 0
                vel_msg_1.angular.z = 0
                self.robot_1_velocity_publisher.publish(vel_msg_1)
                vel_msg_2.linear.x = 0
                vel_msg_2.angular.z = 0
                self.robot_2_velocity_publisher.publish(vel_msg_2)
 
            self.rate.sleep()



    def husky_push(self):

        time.sleep(2)

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