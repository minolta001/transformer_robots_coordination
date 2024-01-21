#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from math import pow,atan2,sqrt,sin,cos
import numpy as np
import threading
import time
from rl.config import argparser
from scipy.spatial.transform import Rotation

import torch
from rl.policies import get_actor_critic_by_name
from rl.low_level_agent import LowLevelAgent
import gym
from rl.trainer import get_subdiv_space, get_agent_by_name, Trainer
from rl.main import make_log_files
from env.transform_utils import Y_vector_from_quat, X_vector_from_quat, l2_dist, alignment_heading_difference, \
    cos_dist, movement_heading_difference, Y_vector_overlapping
from mpi4py import MPI
import numpy as np
from collections import OrderedDict
import signal
import os
from util.logger import logger
from util.pytorch import get_ckpt_path
import sys
import math

goal1_pos = np.array([4, 1, 0.30996])
goal2_pos = np.array([4, -1, 0.30996])
goal_pos = np.array([4, 0, 0.30996])
goal_quat = np.array([1, 0, 0, 0])
box_height = 0.30996
robot_height = 0.15486868

def forward_kinematic(ac_L, ac_R):
        linear_vel = (ac_L + ac_R)
        angular_vel = (ac_R - ac_L) / 3.63636363     # the coefficient is tested on mujoco. We want to reduce the gap between
                                                     # simulation and reality, so let reality yield to the simulation
        return linear_vel, angular_vel 

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


class collab_push():
    def __init__(self, args):

        self.config = args

        rospy.init_node('collab_robots_controller')

        # Husky: pose, linear veloctiy, yaw velocity, quat, forward vector
        self.husky_last_time = None
        self.husky_angular_velocity = [0, 0, 0]
        self.husky_linear_velocity = [0, 0, 0]
        self.husky_last_pose = None
        self.husky_cur_pose = Pose()
        self.husky_quat = [1, 0, 0, 0]      # w, x, y, z
        self.husky_forward_vec = [0, 0, 0]  # forwar vector
        self.husky_right_vec = [0, 0, 0]
        self.husky_pose_vel_subscriber = rospy.Subscriber('/mocap_node/Husky/Odom', Odometry, self.husky_pose_vel_callback)

        # Bunker: pose, linear veloctiy, yaw velocity, quat, forward vector
        self.bunker_last_time = None
        self.bunker_angular_velocity = [0, 0, 0]
        self.bunker_linear_velocity = [0, 0, 0]
        self.bunker_last_pose = None
        self.bunker_cur_pose = Pose()
        self.bunker_quat = [1, 0, 0, 0]    # w, x, y, z
        self.bunker_forward_vec = [0, 0, 0] # forward vector
        self.bunker_right_vec = [0, 0, 0]
        self.bunker_pose_vel_subscriber = rospy.Subscriber('/mocap_node/Bunker/Odom', Odometry, self.bunker_pose_vel_callback)

        # get Box1 pose, quat, forward vector
        self.box1_pose = Pose()
        self.box1_quat = [1, 0, 0, 0]
        self.box1_forward_vec = [0, 0, 0]
        self.box_1_pose_subscriber = rospy.Subscriber('/mocap_node/Box1/Odom', Odometry, self.box1_pose_callback)

        # get Box2 pose, quat, forward vector
        self.box2_pose = Pose()
        self.box2_quat = [1, 0, 0, 0]
        self.box2_forward_vec = [0, 0, 0]
        self.box_2_pose_subscriber = rospy.Subscriber('/mocap_node/Box2/Odom', Odometry, self.box2_pose_callback)

        # get Box pose
        self.box_last_time = None
        self.box_pose = Pose()
        self.box_last_pose = None
        self.box_quat = [1, 0, 0, 0]
        self.box_forward_vec = [0, 0, 0]
        self.box_linear_velocity = [0, 0, 0] # [x, y, z] z is always in 0 in our 2D plane
        self.box_angular_velocity = [0, 0, 0]   # rad


        # huskys velocity publisher 
        self.bunker_velocity_publisher = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10) # publish Twist message to Husky

        # bunker velocity publisher
        self.robot_2_velocity_publisher = rospy.Publisher('/smoother_cmd_vel', Twist, queue_size=10) # publish Twist message to Bunker



        self.rate = rospy.Rate(20)

        #self.trainer = self.load_trainer(args)
        self._test()


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

    def bunker_pose_vel_callback(self, data):
        cur_time = data.header.stamp.secs + (data.header.stamp.nsecs / 1000000000)
        self.bunker_cur_pose = data.pose.pose
        self.bunker_quat = [self.bunker_cur_pose.orientation.w,
                            self.bunker_cur_pose.orientation.x,
                            self.bunker_cur_pose.orientation.y,
                            self.bunker_cur_pose.orientation.z]
        
        self.bunker_forward_vec = X_vector_from_quat(self.bunker_quat)
        self.bunker_right_vec = Y_vector_from_quat(self.bunker_quat)

        if self.bunker_last_pose != None and self.last_time != None:
            delta_linear_x = self.bunker_cur_pose.position.x - self.bunker_last_pose.position.x
            delta_linear_y = self.bunker_cur_pose.position.y - self.bunker_last_pose.position.y

            _, _, cur_yaw_z = euler_from_quaternion(self.bunker_cur_pose.orientation.x,
                                                        self.bunker_cur_pose.orientation.y,
                                                        self.bunker_cur_pose.orientation.z,
                                                        self.bunker_cur_pose.orientation.w)

            _, _, last_yaw_z = euler_from_quaternion(self.bunker_last_pose.orientation.x,
                                                        self.bunker_last_pose.orientation.y,
                                                        self.bunker_last_pose.orientation.z,
                                                        self.bunker_last_pose.orientation.w)


            delta_yaw = cur_yaw_z - last_yaw_z

            time_gap = cur_time - self.bunker_last_time

            if time_gap != 0:
                # linear velocity
                linear_x_velocity = delta_linear_x / time_gap
                linear_y_velocity = delta_linear_y / time_gap

                self.bunker_linear_velocity = [linear_x_velocity, linear_y_velocity, 0]

                # angular velocity
                self.bunker_angular_velocity = [0, 0, delta_yaw / time_gap]

        self.last_time = cur_time
        self.bunker_last_pose = self.bunker_cur_pose

    
    def husky_pose_vel_callback(self, data):
        cur_time = data.header.stamp.secs + (data.header.stamp.nsecs / 1000000000)
        self.husky_cur_pose = data.pose.pose

        self.husky_quat = [self.husky_cur_pose.orientation.w,
                            self.husky_cur_pose.orientation.x,
                            self.husky_cur_pose.orientation.y,
                            self.husky_cur_pose.orientation.z]
        self.husky_forward_vec = X_vector_from_quat(self.husky_quat)
        self.husky_right_vec = Y_vector_from_quat(self.bunker_quat)

        if self.husky_last_pose != None and self.husky_last_time != None:
            delta_linear_x = self.husky_cur_pose.position.x - self.husky_last_pose.position.x
            delta_linear_y = self.husky_cur_pose.position.y - self.husky_last_pose.position.y

            _, _, cur_yaw_z = euler_from_quaternion(self.husky_cur_pose.orientation.x,
                                                        self.husky_cur_pose.orientation.y,
                                                        self.husky_cur_pose.orientation.z,
                                                        self.husky_cur_pose.orientation.w)

            _, _, last_yaw_z = euler_from_quaternion(self.husky_last_pose.orientation.x,
                                                        self.husky_last_pose.orientation.y,
                                                        self.husky_last_pose.orientation.z,
                                                        self.husky_last_pose.orientation.w)


            delta_yaw = cur_yaw_z - last_yaw_z

            time_gap = cur_time - self.husky_last_time

            if time_gap != 0:
                # linear velocity
                linear_x_velocity = delta_linear_x / time_gap
                linear_y_velocity = delta_linear_y / time_gap

                self.husky_linear_velocity = [linear_x_velocity, linear_y_velocity, 0]

                # angular velocity
                self.husky_angular_velocity = [0, 0, delta_yaw / time_gap]
                #print("{0:.8f}".format(self.husky_yaw_velocity))


        self.husky_last_time = cur_time
        self.husky_last_pose = self.husky_cur_pose


    def box_pose_vel_callback(self, data):
        cur_time = data.header.stamp.secs + (data.header.stamp.nsecs / 1000000000)
        self.box_cur_pose = data.pose.pose

        self.box_quat = [self.box_cur_pose.orientation.w,
                            self.box_cur_pose.orientation.x,
                            self.box_cur_pose.orientation.y,
                            self.box_cur_pose.orientation.z]
        self.box_forward_vec = X_vector_from_quat(self.box_quat)

        if self.box_last_pose != None and self.box_last_time != None:
            delta_linear_x = self.box_cur_pose.position.x - self.box_last_pose.position.x
            delta_linear_y = self.box_cur_pose.position.y - self.box_last_pose.position.y

            _, _, cur_yaw_z = euler_from_quaternion(self.box_cur_pose.orientation.x,
                                                        self.box_cur_pose.orientation.y,
                                                        self.box_cur_pose.orientation.z,
                                                        self.box_cur_pose.orientation.w)

            _, _, last_yaw_z = euler_from_quaternion(self.box_last_pose.orientation.x,
                                                        self.box_last_pose.orientation.y,
                                                        self.box_last_pose.orientation.z,
                                                        self.box_last_pose.orientation.w)


            delta_yaw = cur_yaw_z - last_yaw_z

            time_gap = cur_time - self.box_last_time

            if time_gap != 0:
                # linear velocity
                linear_x_velocity = delta_linear_x / time_gap
                linear_y_velocity = delta_linear_y / time_gap

                self.box_linear_velocity = [linear_x_velocity, linear_y_velocity, 0]

                # z-axis angular velocity
                self.box_angular_velocity = [0, 0, delta_yaw / time_gap]
         
        self.box_last_time = cur_time
        self.box_last_pose = self.box_cur_pose


    def box1_pose_callback(self, data):
        self.box1_pose = data.pose.pose
        self.box1_quat = [self.box1_pose.orientation.w,
                          self.box1_pose.orientation.x,
                          self.box1_pose.orientation.y,
                          self.box1_pose.orientation.z]
        self.box1_forward_vec = X_vector_from_quat(self.box1_quat)

    
    def box2_pose_callback(self, data):
        self.box2_pose = data.pose.pose
        self.box2_quat = [self.box2_pose.orientation.w,
                          self.box2_pose.orientation.x,
                          self.box2_pose.orientation.y,
                          self.box2_pose.orientation.z]
        self.box2_forward_vec = X_vector_from_quat(self.box2_quat)

    # relative info of box1-husky, box1-goal1
    def box1_rela_info(self):
        box_pos = np.array([self.box1_pose.position.x, self.box1_pose.position.y, box_height])
        husky_pos = np.array([self.husky_cur_pose.position.x, self.husky_cur_pose.position.y, robot_height])
        
        box1_husky_rela_pos = box_pos - husky_pos
        box1_goal_rela_pos = goal1_pos - box_pos
        husky_move_coeff = movement_heading_difference(box_pos, husky_pos, self.husky_forward_vec, "forward")
        
        rela_info = OrderedDict([
            ('box1_husky_rela_pos', box1_husky_rela_pos),
            ('box1_goal_rela_pos', box1_goal_rela_pos),
            ('box1_husky_move_coeff', husky_move_coeff)
        ])

        return rela_info

    # relative info of box2-bunker, box2-goal2
    def box2_rela_info(self):
        box_pos = np.array([self.box2_pose.position.x, self.box2_pose.position.y, box_height])
        bunker_pos = np.array([self.bunker_cur_pose.position.x, self.bunker_cur_pose.position.y, robot_height])
        
        box2_bunker_rela_pos = box_pos - bunker_pos
        box2_goal_rela_pos = goal2_pos - box_pos
        bunker_move_coeff = movement_heading_difference(box_pos, bunker_pos, self.bunker_forward_vec, "forward")
        
        rela_info = OrderedDict([
            ('box2_bunker_rela_pos', box2_bunker_rela_pos),
            ('box2_goal_rela_pos', box2_goal_rela_pos),
            ('box2_bunker_move_coeff', bunker_move_coeff)
        ])

        return rela_info
    

    # relative info of robots, and box-goal
    def global_rela_info(self):
        husky_pos = np.array([self.husky_cur_pose.position.x, self.husky_cur_pose.position.y, robot_height])
        bunker_pos = np.array([self.bunker_cur_pose.position.x, self.bunker_cur_pose.position.y, robot_height])
                
        goal_forward_vec = X_vector_from_quat(goal_quat)
    
        robots_forward_align_coeff = alignment_heading_difference(self.husky_forward_vec, self.bunker_forward_vec)
        robots_right_align_coeff = Y_vector_overlapping(self.husky_right_vec, self.bunker_right_vec, husky_pos, bunker_pos)
        robots_dist = l2_dist(husky_pos, bunker_pos)
        goal_box_cos_dist = 1 - cos_dist(self.box_forward_vec, goal_forward_vec)
        
        rela_info = OrderedDict([
            ('robots_forward_align_coeff', robots_forward_align_coeff),
            ('robots_right_align_coeff', robots_right_align_coeff),
            ('robots_dist', robots_dist),
            ('goal_box_cos_dist', goal_box_cos_dist)
        ])
         
        return rela_info
    
    # generate observation that could fit the model format
    def make_obs(self):
        box1_rela_info = self.box1_rela_info()
        box2_rela_info = self.box2_rela_info()
        global_rela_info = self.global_rela_info()

        husky_vel = self.husky_linear_velocity + self.husky_angular_velocity
        bunker_vel = self.bunker_linear_velocity + self.bunker_angular_velocity
        box_vel = self.box_linear_velocity + self.box_angular_velocity

        obs = OrderedDict([
            ('husky_1', np.concatenate([[robot_height], self.husky_quat, husky_vel, self.husky_forward_vec, [box1_rela_info['box1_husky_move_coeff']]])),   # husky
            ('husky_2', np.concatenate([[robot_height], self.bunker_quat, bunker_vel, self.bunker_forward_vec, [box2_rela_info['box2_bunker_move_coeff']]])),   # actually bunker!
            ('box_1', np.concatenate([box1_rela_info['box1_husky_rela_pos'], box1_rela_info['box1_goal_rela_pos'], self.box1_forward_vec, box_vel])),
            ('box_2', np.concatenate([box2_rela_info['box2_bunker_rela_pos'], box2_rela_info['box2_goal_rela_pos'], self.box2_forward_vec ,box_vel])),
            ('relative_info', [global_rela_info['robots_forward_align_coeff'], global_rela_info['robots_right_align_coeff'], global_rela_info['robots_dist'], global_rela_info['goal_box_cos_dist']])
        ])

        return obs


    def _test(self):
        print("!!The test is going to start, be careful!!")
        input("Press Enter to start")

        robots_pushing_model = self.load_trainer(self.config)
        
        while not rospy.is_shutdown():
            husky_vel_msg = Twist()
            bunker_vel_msg = Twist()
            obs = self.make_obs()

            meta_ac, meta_ac_before_activation, meta_log_prob = robots_pushing_model._meta_agent.act(obs, is_train=False)

            ll_ob = obs.copy()

            ac, ac_before_activation = robots_pushing_model._agent.act(ll_ob, meta_ac, is_train=False)

            husky_action = ac['husky_1']
            bunker_action = ac['husky_2']

            # husky: transfer action to linear and angular command
            husky_linear_vel, husky_angular_vel = forward_kinematic(husky_action[0], husky_action[1])

            # bunker: transfer action to linear and angular command
            bunker_linear_vel, bunker_angular_vel = forward_kinematic(bunker_action[0], bunker_action[1])

            


    
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