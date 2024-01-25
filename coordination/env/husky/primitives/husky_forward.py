from collections import OrderedDict

import numpy as np

from env.husky.husky import HuskyEnv

from env.transform_utils import up_vector_from_quat, forward_vector_from_quat, \
    l2_dist, cos_dist, right_vector_from_quat, sample_quat, rotate_direction, \
    alignment_heading_difference, movement_heading_difference, forward_backward

import mujoco_py
import math


'''
    Primitve skill:
        1. approach: husky approach to the center point of the box
            - rewards: husky linear velocity, distance toward the box, moving direction
        2. push: husky push the box to move
            - rewards: box linear velocity, box angular velocity
        3. align: husky align its heading to the box
            - rewards: alignment
'''

class HuskyForwardEnv(HuskyEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_forward'
        super().__init__('husky_forward.xml', **kwargs)

        # Env config
        self._env_config.update({
            'dist_reward': 30,
            'linear_vel_reward': 50, #50 TODO
            'angular_vel_reward': 20,   # TODO
            #'box_linear_vel_reward': 5000,
            'box_linear_vel_reward': 100,
            'box_angular_vel_reward': 20,
            'box_goal_reward': 500,
            'alive_reward': 0.,
            'quat_reward': 10, # 0
            'align_move_both_reward': 10,
            'die_penalty': 500,
            'max_episode_steps': 200, #was 500, will shorter episode be better?
            'husky': 1,
            'skill': 'approach',
            'init_randomness': 0.1,
            'diayn_reward': 5,    # was 0.1
            "prob_perturb_action": 0.1,    #0.1
            "perturb_action": 0.01,
            "alignment_reward": 10,
            "move_heading_reward": 50,
            "bonus_reward": 5
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        # Env info
        self.husky = "husky_%d" % self._env_config['husky']
        self.box = "box_%d" % self._env_config['husky']

        '''
            here define the size of observation space
            husky has 29 in total:
                9 from qpos (body and 4 wheels)
                10 from qvel
                10 from qacc

            box has 6 in total:
                3 from pos difference between box and husky
                3 from forward vecto_perturb_actionr

        self.ob_shape = OrderedDict([(self.husky, 30), 
                                     (self.box, 9),
                                     ])
        '''


        '''
        2023/12/08 Current size of observation
        '''
        self.ob_shape = OrderedDict([(self.husky, 15),
                                     (self.box, 12)])


        '''
            Our husky model is differential drive, though it has 4 wheels.
            So, the action space should be 2
        '''
        self.action_space.decompose(OrderedDict([(self.husky, 2)]))


    def _step(self, a):
        pos_before = self._get_pos('husky_geom')
        box_before = self._get_pos('box_geom')


        '''
        NOTE: THIS IS VERY IMPORTANT!

        Use right_vector_from_quat to acquire forward vector of object and robots. The original forward_vector_from_quat takes the world map as the coordination, so y-axis is the actual forward direction.
        '''

        husky_quat_before = self._get_quat('husky_robot')
        husky_forward_vector_before = right_vector_from_quat(husky_quat_before)
        
        box_quat_before = self._get_quat('box')
        box_forward_vector_before = right_vector_from_quat(box_quat_before)
        a = self._perturb_action(a)

        # Do a simulation
        self.do_simulation(a)

        #self.husky_simulation(a)
        pos_after = self._get_pos('husky_geom')
        box_after = self._get_pos('box_geom')
        goal_pos_after = self._get_pos('goal_geom')
        box_quat_after = self._get_quat('box')
        goal_quat_after = self._get_quat('goal')
        

        husky_quat_after = self._get_quat('husky_robot')
        husky_forward_vector_after = right_vector_from_quat(husky_quat_after) 

        # box orientation on z-axis     # box forward vector after
        box_forward = right_vector_from_quat(box_quat_after)
        # box angular velocity
        box_angular_vel = cos_dist(self._box_forward, box_forward)
        # goal forward vector
        goal_forward = right_vector_from_quat(goal_quat_after)


        ob = self._get_obs()
        done = False
        alive_reward = 0
        #ctrl_reward = self._ctrl_reward(a)
        ctrl_reward = self._ctrl_reward(a) * 10
        

        '''
            Distance Calculation
        '''
        # distance between husky and center point of the box 
        dist_husky_box = l2_dist(pos_after, box_after)

        #dist_husky_box_reward = (2 - dist_husky_box) * self._env_config["dist_reward"]

        ########## this is quadratic potential field reward
        # attractive force to husky_box dist reward
        dist_husky_box_reward = 2 / dist_husky_box * self._env_config["dist_reward"]

        # distance between box and goal
        dist_box_goal = l2_dist(goal_pos_after, box_after)
        #dist_box_goal_reward = (2 - dist_box_goal) * self._env_config["dist_reward"]
        dist_box_goal_reward =  4 / dist_box_goal * self._env_config["dist_reward"]

        # quat distance between box and goal
        # TODO: Unused
        quat_dist_box_goal = cos_dist(box_forward, goal_forward)
        quat_dist_box_goal_reward = (1 - quat_dist_box_goal) * self._env_config["quat_reward"]



        '''
            Velocity Calculation
        ''' 
        #husky_linear_vel = abs(pos_after[0] - pos_before[0]) + abs(pos_after[1] - pos_before[1])
        husky_linear_vel = l2_dist(pos_after, pos_before)
        husky_move_direction = forward_backward(self.data.qvel.ravel().copy())
        # husky angular velocity 
        husky_angular_vel = cos_dist(husky_forward_vector_before, husky_forward_vector_after)
        # box linear velocity
        box_linear_vel = l2_dist(box_after, box_before)


        # distance between box and husky
        box_dist = box_after - pos_after

        # reward components, for later use
        # TODO: these three are Unused!
        husky_linear_vel_reward = self._env_config["linear_vel_reward"] * husky_linear_vel * 10
        husky_angular_vel_reward = self._env_config["angular_vel_reward"] * husky_angular_vel
        box_angular_vel_reward = self._env_config["box_angular_vel_reward"] * (1- box_angular_vel)


        # attractive force to Box Linear Reward
        #box_linear_vel_reward = self._env_config["box_linear_vel_reward"] * (box_linear_vel)
        #box_linear_vel_reward = self._env_config["linear_vel_reward"] * (1 / abs(dist_box_goal - box_linear_vel * 5))
        
        box_linear_vel_reward = -self._env_config["linear_vel_reward"] * abs(dist_box_goal - box_linear_vel)
        

        alive_reward = self._env_config["alive_reward"]

        # If robot's heading is parallel with object's heading?
        align_coeff, direction = alignment_heading_difference(box_forward, husky_forward_vector_after)


        #alignment_heading_reward = self._env_config["alignment_reward"] * align_coeff

        # if robot is moving toward the object?
        move_coeff = 0
        movement_heading_reward = 0
        # checking if heading forwards to the object
        move_coeff = movement_heading_difference(box_after, 
                                                    pos_after, 
                                                    husky_forward_vector_after, 
                                                    "forward")

        '''
            Failure Check
        '''
        if not np.isfinite(self.data.qpos).all():
            done = True

        # if the husky is too far away from the husky
        if abs(box_dist[0]) > 5 or abs(box_dist[1]) > 5:
            done = True

        if dist_box_goal > 5:
            done = True

        # if the husky heading is wrong
        if move_coeff < 0.3:
            done = True        
        
        die_penalty = -self._env_config["die_penalty"] if done else 0
        
        
        '''
            Reward
        '''
        # Different reward functions for different primitive skills
        skill = self._env_config["skill"]
        rotate_direct = rotate_direction(husky_forward_vector_before, husky_forward_vector_after)


        '''
            Adding up reward
        '''
        reward = ctrl_reward + alive_reward + die_penalty


        if (skill == "push"):
            both_align = 0  # 1 if both alignment and moving direction are closed to correct

            #if dist_husky_box < 1.2:    # husky is getting into a push-ready range
            reward = reward + self._env_config['bonus_reward']
            #reward = reward + align_coeff * self._env_config['alignment_reward']
            #reward = reward + movement_heading_reward

            '''
            if dist_box_goal < 0.4 and husky_linear_vel < 0.005:
                reward = reward + 500
            '''

            if dist_box_goal < 0.18:
                reward = reward + self._env_config['box_goal_reward']
                done = True

            if dist_box_goal > 5:   # Fail
                done = True
                

            move_coeff = movement_heading_difference(box_after, 
                                                     pos_after, 
                                                     husky_forward_vector_after, 
                                                     "forward")
            movement_heading_reward = self._env_config['move_heading_reward'] * move_coeff

            reward = reward \
                    + dist_husky_box_reward \
                    + dist_box_goal_reward \
                    + box_linear_vel_reward \
                    #+ quat_dist_box_goal_reward \
                    #+ box_angular_vel_reward

        if (skill == "move"):
            move_coeff = movement_heading_difference(box_after, 
                                                     pos_after, 
                                                     husky_forward_vector_after, 
                                                     "forward")
            
            dist_husky_box_before = l2_dist(box_before, pos_before)
            dist_husky_box_after = l2_dist(box_after, pos_after)
            dist_diff = dist_husky_box_before - dist_husky_box_after
            move_direction = np.sign(dist_diff)     # if robot move toward box, the sign is positive, so we reward the robot's moving
        
            husky_linear_vel_reward = husky_linear_vel * move_coeff * self._env_config['linear_vel_reward'] * 5 * move_direction
            box_linear_vel_reward = box_linear_vel * self._env_config['box_linear_vel_reward'] * 10
            dist_reward = l2_dist(box_after, np.array([0, 0, 0.3])) * self._env_config['dist_reward']
            dist_husky_box_reward = dist_husky_box_reward / 10
            reward = reward + husky_linear_vel_reward + box_linear_vel_reward + dist_reward + dist_husky_box_reward 
        
        elif(skill == "approach"):  # encourage the robot push slowly when the object is near the goal
            reward = reward + self._env_config['bonus_reward']
            
            if dist_box_goal > 2.06:   # box is far away from the goal. Fail.
                reward = reward - self._env_config['box_goal_reward']
                done = True
            
            move_coeff = movement_heading_difference(box_after, 
                                                     pos_after, 
                                                     husky_forward_vector_after, 
                                                     "forward")
            movement_heading_reward = self._env_config['move_heading_reward'] * move_coeff * 3
            reward = reward \
                    + dist_husky_box_reward \
                    + movement_heading_reward \


        self._reward = reward

        env_config = self._env_config 

        info = {"Current Skill": skill,
                "Total Reward": reward,
                "reward: dist_husky_box_reward": dist_husky_box_reward,
                "reward: husky_linear": husky_linear_vel_reward,
                "reward: husky_angular": husky_angular_vel_reward,
                "reward: movingment_heading": movement_heading_reward,
                #"reward: heading_alignment": alignment_heading_reward,
                "reward: box_linear_reward": box_linear_vel_reward,
                "reward: box_angular": box_angular_vel_reward,
                "----------": 0,
                "husky_forward or backward": husky_move_direction,
                "husky_movement_direction_coeff": move_coeff,
                "husky_alignment_coeff": align_coeff,
                "----------": 0,
                "----------": 0,
                "penalty_die": die_penalty,
                "box_forward": box_forward,
                "dist_husky_box": dist_husky_box,
                "dist_husky_readypos:": abs(dist_husky_box - 1.5),
                "husky_pos": pos_after,
                "box_pos": box_after,
                "box_ob": ob[self.box], 
                "success": self._success
            }

        if (skill == "push"):
            info = {"Current Skill": skill,
                "Total Reward": reward,
                "reward: dist_husky_box": dist_husky_box_reward,
                "reward: dist_box_goal": dist_box_goal_reward,
                "reward: box_linear_vel": box_linear_vel_reward,
                "reward: control reward": ctrl_reward,
                #"reward: heading_alignment": alignment_heading_reward,
                "----------": 0,
                "husky_movement_heading_coeff": move_coeff,
                #"husky_alignment_coeff": align_coeff,
                #"align_move_both_>0.95": both_align,
                "------------": 0,
                "dist_husky_box": dist_husky_box,
                "dist_box_goal": dist_box_goal,
                "husky_pos": pos_after,
                "box_pos": box_after,
                "box_ob": ob[self.box], 
                "success": self._success
            }

        if (skill == "move"):
            info = {"Current Skill": skill,
                    "Total Reward": reward,
                    "reward: box_linear_vel": box_linear_vel_reward,
                    "reward: husky_linear_vel": husky_linear_vel_reward,
                    "reward: box move distance reward": dist_reward,
                    "reward: husky box distance reward": dist_husky_box_reward,
                    "----------": 0,
                    "husky_movement_heading_coeff": move_coeff,
                    "----------": 0,
                    "dist_box_goal": dist_box_goal,
                    "husky_pos": pos_after,
                    "husky_quat": husky_quat_after,
                    "box_pos": box_after,
                    "box_ob": ob[self.box], 
                    "success": self._success
            }
        if (skill == "approach"):
            info = {"Current Skill": skill,
                    "Total Reward": reward,
                    "reward: dist_husky_box": dist_husky_box_reward,
                    #"reward: box_linear_vel": box_linear_vel_reward,
                    "reward: moving heading": movement_heading_reward,
                    #"reward: husky_linear_vel": husky_linear_vel_reward,
                    "----------": 0,
                    "husky_movement_heading_coeff": move_coeff,
                    "----------": 0,
                    "dist_husky_box": dist_husky_box,
                    "dist_box_goal": dist_box_goal,
                    "husky_pos": pos_after,
                    "box_pos": box_after,
                    "box_ob": ob[self.box], 
                    "success": self._success
            }

        return ob, reward, done, info

    def _get_obs(self):
        # Husky
        qpos = self.data.qpos
        qvel = self.data.qvel

        qacc = self.data.qacc

        husky_pos = self._get_pos('husky_robot')
        # box
        box_pos = self._get_pos('box_geom')
        #box_quat = self._get_quat('box')
        box_forward = self._get_right_vector('box_geom')

        # goal
        goal_pos = self._get_pos('goal_geom')

        # relative
        husky_forward_vec = right_vector_from_quat(self._get_quat("husky_robot"))
        move_coeff = movement_heading_difference(box_pos, husky_pos, husky_forward_vec, "forward")
        align_coeff, direction = alignment_heading_difference(box_forward, husky_forward_vec)
    

        # You should be able to find the observation space in this scenario
        # Check _reset below to see what are in qpos
        # The same way could be also applied to qvel and qacc
        '''
        obs = OrderedDict([
            (self.husky, np.concatenate([qpos[4:11], qvel[:10], qacc[:10], husky_forward_vec])),
            (self.box, np.concatenate([box_pos - husky_pos, goal_pos - box_pos, box_forward])),
            #('shared_pos', np.concatenate([qpos[2:7], qvel[:6], qacc[:6]])),
            #('lower_body', np.concatenate([qpos[7:15], qvel[6:14], qacc[6:14]])),
        ])
        '''


        '''
        Husky: height, quaternion, linear and angular velocity of robot
        '''
        husky_pos = self._get_pos("husky_robot")
        husky_quat = self._get_quat("husky_robot")
        husky_vel = self._get_vel("husky_body")

        box_vel = self._get_vel("box_body")

        '''
        obs = OrderedDict([
            (self.husky, np.concatenate([husky_pos[2:3], husky_quat, husky_vel, husky_forward_vec, [move_coeff]])), # 15
            (self.box, np.concatenate([box_pos - husky_pos, goal_pos - box_pos, box_forward, box_vel]))      # 15
        ])
        '''

        obs = OrderedDict([
            (self.husky, np.concatenate([husky_pos[2:3], husky_quat, husky_vel, husky_forward_vec, [move_coeff]])), # 15
            (self.box, np.concatenate([box_pos - husky_pos, box_forward, box_vel]))      # 12
        ])

        
        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    @property
    def _init_qpos(self):
        # 3 for (x, y, z), 4 for (x, y, z, w), and 2 for each leg
        '''
        return np.array([0., 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         20., -1., 0.8, 1., 0., 0., 0.])
        '''

        return np.array([-2., 0., 0.218, 0., 0., 0., 0., 0., 0., 0., 0., 
                         0., 0., 0., 0., 0., 0., 0.])

    @property 
    def _init_qvel(self):
        return np.zeros(self.model.nv)
    
    def _reset(self):
        #   IMPORTANT !!
        '''
            total number of joint: 18
            The first 7 are husky_robot qpos
            Then, follow by 4 qpos of 4 wheels
            0 - 6: husky_robot qpos
            7 - 10: wheel qpos
            11 - 13: pos of box_geom
            14 - 17: quaterion of 
        ''' 
        # total number of velocity: 16

        #qpos = self._init_qpos + self._init_random(self.model.nq)
        qpos = self._init_qpos


        #qvel = self._init_qvel + self._init_random(self.model.nv)
        qvel = self._init_qvel

        # DON'T PUT HUSKY TOO HIGH!
        qpos[2] = 0.218
        qpos[3] = 0
        qpos[4] = 0
        qpos[5] = 0
        
        #self.set_state(qpos, qvel)

        self._reset_husky_box()

        return self._get_obs()
    
    def _reset_husky_box(self):

        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # Initialized Husky
        x = np.random.uniform(low=-2, high=-0.8)
        y = np.random.uniform(low=-0.5, high=0.5)
        

        # Initialize box position 
        init_box_pos = np.asarray([0, 0, 0.3])
        init_box_quat = np.array([0, 0, 0, 0])

        if(self._env_config["skill"] == "push" or self._env_config["skill"] == "move"):
            qpos[0] = x
            qpos[1] = y

            husky_quat = sample_quat(low=-np.pi/3, high=np.pi/3)
            self._set_quat('husky_robot', husky_quat)

            # reset the rotation of goal
            
            #goal_pos = np.asarray([1, 0, 0.3])

            goal_pos = np.asarray([np.random.uniform(low=0.5, high=3), np.random.uniform(low=-2, high=2), 0.3])
            goal_quat = sample_quat(low=-np.pi/6, high=np.pi/6)
            self._set_pos('goal', goal_pos)
            self._set_quat('goal', goal_quat)

            # reset the box, reset pos and quat respectively
            init_box_quat = sample_quat(low=-np.pi/6, high=np.pi/6)
            qpos[11:14] = init_box_pos
            qpos[14:18] = init_box_quat
            self._set_pos('box', init_box_pos)
            self._set_quat('box', init_box_quat)
            self.set_state(qpos, qvel)

        elif(self._env_config["skill"] == "approach"):
            qpos[0] = x
            qpos[1] = y
            self._set_quat('husky_robot', sample_quat(low=-np.pi/6, high=np.pi/6))

            # reset goal pose
            goal_pos = np.asarray([np.random.uniform(low=-0.5, high=0.5), np.random.uniform(low=-2, high=2), 0.3])
            goal_quat = sample_quat(low=-np.pi/6, high=np.pi/6)
            self._set_pos('goal', goal_pos)
            self._set_quat('goal', goal_quat)

            # rest the box, reset pos and quat respectively 
            qpos[11:14] = init_box_pos
            qpos[14:18] = init_box_quat

        
        if(self._env_config["skill"] == "wait"):
            self._set_pos('husky_robot', np.array([x, y, 0.19]))
            self._set_quat('husky_robot', sample_quat(low=-np.pi/2, high=np.pi/2)) 

            goal_pos = np.asarray([1, 0, 0.3])
            goal_quat = sample_quat(low=-np.pi/9, high=np.pi/9)
            self._set_pos('goal', goal_pos)
            self._set_quat('goal', goal_quat)
            self._set_pos('box', goal_pos)
            self._set_quat('box', goal_quat)

            qpos[11:14] = goal_pos
            qpos[14:18] = goal_quat




        self._box_forward = right_vector_from_quat(init_box_quat)

        self._husky_pos = (qpos[0], qpos[1])

        #self.set_state(qpos, qvel)




        

    
    
    



