#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import numpy as np
import threading
import time
from rl.config import argparser
from scipy.spatial.transform import Rotation

import torch
from rl.trainer import Trainer
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

# straight
#goal1_pos = np.array([2.5, 1, 0.30996])
#goal2_pos = np.array([2.5, -1, 0.30996])
#goal_pos = np.array([2.5, 0, 0.30996])
#goal_quat = np.array([1, 0, 0, 0])
# curve
goal1_pos = np.array([1.2328, 1.1607, 0.30996])
goal2_pos = np.array([1.9134, -0.7330, 0.30996])
goal_pos = np.array([1.5453, 0.2903, 0.30996])
goal_quat = np.array([0.9847, 0, 0, 0.1737])
box_height = 0.30996
robot_height = 0.15486868

control_linear_slowdown_coeff = 8
control_angular_slowdown_coeff = 1


def forward_kinematic(ac_L, ac_R):
        linear_vel = (ac_L + ac_R) / 2 / control_linear_slowdown_coeff
        angular_vel = (ac_R - ac_L) / 3.63636363 / control_angular_slowdown_coeff   # the coefficient is tested on mujoco. We want to reduce the gap between
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

        # jackal_1: pose, linear veloctiy, yaw velocity, quat, forward vector
        self.jackal_1_last_time = None
        self.jackal_1_angular_velocity = [0, 0, 0]
        self.jackal_1_linear_velocity = [0, 0, 0]
        self.jackal_1_last_pose = None
        self.jackal_1_cur_pose = Pose()
        self.jackal_1_quat = [1, 0, 0, 0]      # w, x, y, z
        self.jackal_1_forward_vec = [0, 0, 0]  # forwar vector
        self.jackal_1_right_vec = [0, 0, 0]
        self.jackal_1_pose_vel_subscriber = rospy.Subscriber('/mocap_node/jackal_1/Odom', Odometry, self.jackal_1_pose_vel_callback)

        # jackal_2: pose, linear veloctiy, yaw velocity, quat, forward vector
        self.jackal_2_last_time = None
        self.jackal_2_angular_velocity = [0, 0, 0]
        self.jackal_2_linear_velocity = [0, 0, 0]
        self.jackal_2_last_pose = None
        self.jackal_2_cur_pose = Pose()
        self.jackal_2_quat = [1, 0, 0, 0]    # w, x, y, z
        self.jackal_2_forward_vec = [0, 0, 0] # forward vector
        self.jackal_2_right_vec = [0, 0, 0]
        self.jackal_2_pose_vel_subscriber = rospy.Subscriber('/mocap_node/jackal_2/Odom', Odometry, self.jackal_2_pose_vel_callback)

        # get Box1 pose, quat, forward vector
        self.box1_pose = Pose()
        self.box1_quat = [1, 0, 0, 0]
        self.box1_forward_vec = [0, 0, 0]
        self.box1_goal_dist = math.inf
        self.box_1_pose_subscriber = rospy.Subscriber('/mocap_node/Box1/Odom', Odometry, self.box1_pose_callback)

        # get Box2 pose, quat, forward vector
        self.box2_pose = Pose()
        self.box2_quat = [1, 0, 0, 0]
        self.box2_forward_vec = [0, 0, 0]
        self.box2_goal_dist = math.inf
        self.box_2_pose_subscriber = rospy.Subscriber('/mocap_node/Box2/Odom', Odometry, self.box2_pose_callback)

        # get Box pose
        self.box_last_time = None
        self.box_cur_pose = Pose()
        self.box_last_pose = None
        self.box_quat = [1, 0, 0, 0]
        self.box_forward_vec = [0, 0, 0]
        self.box_linear_velocity = [0, 0, 0] # [x, y, z] z is always in 0 in our 2D plane
        self.box_angular_velocity = [0, 0, 0]   # rad
        self.box_pose_vel_subscriber = rospy.Subscriber('/mocap_node/Box/Odom', Odometry, self.box_pose_vel_callback)

        # jackal_1 velocity publisher 
        self.jackal_1_vel_publisher = rospy.Publisher('/Jackal_1_cmd_vel', Twist, queue_size=10) # publish Twist message tojackal_1 

        # jackal_2 velocity publisher
        self.jackal_2_vel_publisher = rospy.Publisher('/Jackal_2_cmd_vel', Twist, queue_size=10) # publish Twist message tojackal_2 

        self.rate = rospy.Rate(20)


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

    
    def jackal_1_pose_vel_callback(self, data):
        cur_time = data.header.stamp.secs + (data.header.stamp.nsecs / 1000000000)
        self.jackal_1_cur_pose = data.pose.pose

        self.jackal_1_quat = [self.jackal_1_cur_pose.orientation.w,
                            self.jackal_1_cur_pose.orientation.x,
                            self.jackal_1_cur_pose.orientation.y,
                            self.jackal_1_cur_pose.orientation.z]
        self.jackal_1_forward_vec = X_vector_from_quat(self.jackal_1_quat)
        self.jackal_1_right_vec = Y_vector_from_quat(self.jackal_1_quat)

        if self.jackal_1_last_pose != None and self.jackal_1_last_time != None:
            delta_linear_x = self.jackal_1_cur_pose.position.x - self.jackal_1_last_pose.position.x
            delta_linear_y = self.jackal_1_cur_pose.position.y - self.jackal_1_last_pose.position.y

            _, _, cur_yaw_z = euler_from_quaternion(self.jackal_1_cur_pose.orientation.x,
                                                        self.jackal_1_cur_pose.orientation.y,
                                                        self.jackal_1_cur_pose.orientation.z,
                                                        self.jackal_1_cur_pose.orientation.w)

            _, _, last_yaw_z = euler_from_quaternion(self.jackal_1_last_pose.orientation.x,
                                                        self.jackal_1_last_pose.orientation.y,
                                                        self.jackal_1_last_pose.orientation.z,
                                                        self.jackal_1_last_pose.orientation.w)


            delta_yaw = cur_yaw_z - last_yaw_z

            time_gap = cur_time - self.jackal_1_last_time

            if time_gap != 0:
                # linear velocity
                linear_x_velocity = delta_linear_x / time_gap
                linear_y_velocity = delta_linear_y / time_gap

                self.jackal_1_linear_velocity = [linear_x_velocity, linear_y_velocity, 0]

                # angular velocity
                self.jackal_1_angular_velocity = [0, 0, delta_yaw / time_gap]
                #print("{0:.8f}".format(self.jackal_1_yaw_velocity))


        self.jackal_1_last_time = cur_time
        self.jackal_1_last_pose = self.jackal_1_cur_pose


    def jackal_2_pose_vel_callback(self, data):
        cur_time = data.header.stamp.secs + (data.header.stamp.nsecs / 1000000000)
        self.jackal_2_cur_pose = data.pose.pose
        self.jackal_2_quat = [self.jackal_2_cur_pose.orientation.w,
                            self.jackal_2_cur_pose.orientation.x,
                            self.jackal_2_cur_pose.orientation.y,
                            self.jackal_2_cur_pose.orientation.z]
        
        self.jackal_2_forward_vec = X_vector_from_quat(self.jackal_2_quat)
        self.jackal_2_right_vec = Y_vector_from_quat(self.jackal_2_quat)

        if self.jackal_2_last_pose != None and self.jackal_2_last_time != None:
            delta_linear_x = self.jackal_2_cur_pose.position.x - self.jackal_2_last_pose.position.x
            delta_linear_y = self.jackal_2_cur_pose.position.y - self.jackal_2_last_pose.position.y

            _, _, cur_yaw_z = euler_from_quaternion(self.jackal_2_cur_pose.orientation.x,
                                                        self.jackal_2_cur_pose.orientation.y,
                                                        self.jackal_2_cur_pose.orientation.z,
                                                        self.jackal_2_cur_pose.orientation.w)

            _, _, last_yaw_z = euler_from_quaternion(self.jackal_2_last_pose.orientation.x,
                                                        self.jackal_2_last_pose.orientation.y,
                                                        self.jackal_2_last_pose.orientation.z,
                                                        self.jackal_2_last_pose.orientation.w)


            delta_yaw = cur_yaw_z - last_yaw_z

            time_gap = cur_time - self.jackal_2_last_time

            if time_gap != 0:
                # linear velocity
                linear_x_velocity = delta_linear_x / time_gap
                linear_y_velocity = delta_linear_y / time_gap

                self.jackal_2_linear_velocity = [linear_x_velocity, linear_y_velocity, 0]

                # angular velocity
                self.jackal_2_angular_velocity = [0, 0, delta_yaw / time_gap]

        self.last_time = cur_time
        self.jackal_2_last_pose = self.jackal_2_cur_pose


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

    # relative info of box1-jackal_1, box1-goal1
    def box1_rela_info(self):
        box_pos = np.array([self.box1_pose.position.x, self.box1_pose.position.y, box_height])
        jackal_1_pos = np.array([self.jackal_1_cur_pose.position.x, self.jackal_1_cur_pose.position.y, robot_height])
        
        box1_jackal_1_rela_pos = box_pos - jackal_1_pos
        box1_goal_rela_pos = goal1_pos - box_pos
        jackal_1_move_coeff = movement_heading_difference(box_pos, jackal_1_pos, self.jackal_1_forward_vec, "forward")

        self.box1_goal_dist = l2_dist(box_pos, goal1_pos)
        
        rela_info = OrderedDict([
            ('box1_jackal_1_rela_pos', box1_jackal_1_rela_pos),
            ('box1_goal_rela_pos', box1_goal_rela_pos),
            ('box1_jackal_1_move_coeff', jackal_1_move_coeff)
        ])

        return rela_info

    # relative info of box2-jackal_2, box2-goal2
    def box2_rela_info(self):
        box_pos = np.array([self.box2_pose.position.x, self.box2_pose.position.y, box_height])
        jackal_2_pos = np.array([self.jackal_2_cur_pose.position.x, self.jackal_2_cur_pose.position.y, robot_height])
        
        box2_jackal_2_rela_pos = box_pos - jackal_2_pos
        box2_goal_rela_pos = goal2_pos - box_pos
        jackal_2_move_coeff = movement_heading_difference(box_pos, jackal_2_pos, self.jackal_2_forward_vec, "forward") 
    
        self.box2_goal_dist = l2_dist(box_pos, goal2_pos)
        
        rela_info = OrderedDict([
            ('box2_jackal_2_rela_pos', box2_jackal_2_rela_pos),
            ('box2_goal_rela_pos', box2_goal_rela_pos),
            ('box2_jackal_2_move_coeff', jackal_2_move_coeff)
        ])

        return rela_info
    

    # relative info of robots, and box-goal
    def global_rela_info(self):
        jackal_1_pos = np.array([self.jackal_1_cur_pose.position.x, self.jackal_1_cur_pose.position.y, robot_height])
        jackal_2_pos = np.array([self.jackal_2_cur_pose.position.x, self.jackal_2_cur_pose.position.y, robot_height])
                
        goal_forward_vec = X_vector_from_quat(goal_quat)
    
        robots_forward_align_coeff = alignment_heading_difference(self.jackal_1_forward_vec, self.jackal_2_forward_vec)
        robots_right_align_coeff = Y_vector_overlapping(self.jackal_1_right_vec, self.jackal_2_right_vec, jackal_1_pos, jackal_2_pos)
        robots_dist = l2_dist(jackal_1_pos, jackal_2_pos)
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

        jackal_1_vel = self.jackal_1_linear_velocity + self.jackal_1_angular_velocity
        jackal_2_vel = self.jackal_2_linear_velocity + self.jackal_2_angular_velocity
        box_vel = self.box_linear_velocity + self.box_angular_velocity

        obs = OrderedDict([
            ('husky_1', np.concatenate([[robot_height], self.jackal_1_quat, jackal_1_vel, self.jackal_1_forward_vec, [box1_rela_info['box1_jackal_1_move_coeff']]])),   #jackal_1 
            ('husky_2', np.concatenate([[robot_height], self.jackal_2_quat, jackal_2_vel, self.jackal_2_forward_vec, [box2_rela_info['box2_jackal_2_move_coeff']]])),   # actually jackal_2!
            ('box_1', np.concatenate([box1_rela_info['box1_jackal_1_rela_pos'], self.box1_forward_vec, box_vel])),
            ('box_2', np.concatenate([box2_rela_info['box2_jackal_2_rela_pos'],self.box2_forward_vec ,box_vel])),
            ('relative_info', np.concatenate([box1_rela_info['box1_goal_rela_pos'],box2_rela_info['box2_goal_rela_pos'] ,[global_rela_info['robots_forward_align_coeff'], global_rela_info['robots_right_align_coeff'], global_rela_info['robots_dist'], global_rela_info['goal_box_cos_dist']]]))
        ])

        assert(len(obs['husky_1']) == 15)
        assert(len(obs['husky_2']) == 15)
        assert(len(obs['box_1']) == 12)
        assert(len(obs['box_2']) == 12)
        assert(len(obs['relative_info']) == 10)

        return obs


    def _test(self):

        print("Loading collaborative robtos pushing model")

        robots_pushing_model = self.load_trainer(self.config)

        print("!!The test is going to start, be careful!!")
        input("Press Enter to start")
        time.sleep(5)

        while not rospy.is_shutdown():
            jackal_1_pos = np.array([self.jackal_1_cur_pose.position.x, self.jackal_1_cur_pose.position.y, robot_height])
            jackal_2_pos = np.array([self.jackal_2_cur_pose.position.x, self.jackal_2_cur_pose.position.y, robot_height]) 
            jackal_1_x = jackal_1_pos[0]
            jackal_1_y = jackal_1_pos[1]
            jackal_2_x = jackal_2_pos[0]
            jackal_2_y = jackal_2_pos[1]

            box1_pos = self.box1_pose.position
            box2_pos = self.box2_pose.position


            robots_dist = l2_dist(jackal_1_pos, jackal_2_pos)

            jackal_1_vel_msg = Twist()
            jackal_2_vel_msg = Twist()
            obs = self.make_obs()

            meta_ac, meta_ac_before_activation, meta_log_prob = robots_pushing_model._meta_agent.act(obs, is_train=False)

            ll_ob = obs.copy()

            ac, ac_before_activation = robots_pushing_model._agent.act(ll_ob, meta_ac, is_train=False)

            jackal_1_action = ac['husky_1']
            jackal_2_action = ac['husky_2']

            # jackal_1: transfer action to linear and angular command
            jackal_1_linear_vel, jackal_1_angular_vel = forward_kinematic(jackal_1_action[0], jackal_1_action[1])

            # jackal_2: transfer action to linear and angular command
            jackal_2_linear_vel, jackal_2_angular_vel = forward_kinematic(jackal_2_action[0], jackal_2_action[1])
            print(self.box_cur_pose.position)
            print("goal1 dist: ", self.box1_goal_dist, "goal2 dist: ", self.box2_goal_dist)            
            # make sure robots within safety zone, and won't collide together
            # also make sure box won't crash anything
            if(-2.5 <= jackal_1_x and jackal_1_x <= 2.5 and -2 <= jackal_1_y and jackal_1_y <= 1.9 \
               and -2.5 <= jackal_2_x and jackal_2_x <= 2.5 and -2 <= jackal_2_y and jackal_2_y <= 1.9 \
                and 0.5 <= robots_dist and (self.box1_goal_dist > 0.4 or self.box2_goal_dist > 0.4) and
                -2.5 <= box1_pos.x and box1_pos.x <= 2.5 and -2 <= box1_pos.y and box1_pos.y <= 1.9 \
                and -2.5 <= box2_pos.x and box2_pos.x <= 2.5 and -2 <= box2_pos.y and box2_pos.y <= 1.9):

                    jackal_1_vel_msg.linear.x = jackal_1_linear_vel
                    jackal_1_vel_msg.angular.z = jackal_1_angular_vel
                    jackal_2_vel_msg.linear.x = jackal_2_linear_vel
                    jackal_2_vel_msg.angular.z = jackal_2_angular_vel
                    self.jackal_1_vel_publisher.publish(jackal_1_vel_msg)
                    self.jackal_2_vel_publisher.publish(jackal_2_vel_msg)
            else:
                jackal_1_vel_msg.linear.x = 0
                jackal_1_vel_msg.angular.z = 0 
                jackal_2_vel_msg.linear.x = 0 
                jackal_2_vel_msg.angular.z = 0 
                self.jackal_1_vel_publisher.publish(jackal_1_vel_msg)
                self.jackal_2_vel_publisher.publish(jackal_2_vel_msg)

                if(self.box1_goal_dist <= 0.5 and self.box2_goal_dist <= 0.5):
                    print("Goal achieve: ", self.box_cur_pose.position)
                else:
                    print("Some safety distancing is violated!")
                    print("jackal_1:", self.jackal_1_cur_pose.position)
                    print("jackal_2:", self.jackal_2_cur_pose.position)
                    print("box1", self.box1_pose.position)
                    print("box2", self.box2_pose.position)
                
                rospy.on_shutdown()
                
            print("goal1 dist: ", self.box1_goal_dist, "goal2 dist: ", self.box2_goal_dist)
            
    
if __name__ == '__main__':
    args, unparsed = argparser()
    collab_push_team = collab_push(args)
    collab_push_team._test()