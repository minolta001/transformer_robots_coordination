from collections import OrderedDict

import numpy as np

from env.husky.husky import HuskyEnv

from env.transform_utils import up_vector_from_quat, forward_vector_from_quat, \
    l2_dist, cos_dist, right_vector_from_quat, sample_quat, rotate_direction, \
    alignment_heading_difference, movement_heading_difference, forward_backward

import mujoco_py
import math

class HuskyForwardEnv(HuskyEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_forward'
        super().__init__('husky_forward.xml', **kwargs)

        # Env config
        self._env_config.update({
            'linear_vel_reward': 1000, #50
            'angular_vel_reward': 50,
            'box_linear_vel_reward': 5000,
            'box_angular_vel_reward': 50,
            'offset_reward': 1,
            'height_reward': 0.5,
            'upright_reward': 5,
            'alive_reward': 0.,
            'quat_reward': 30, # 0
            'die_penalty': 10,
            'max_episode_steps': 500,
            'husky': 1,
            'direction': 'right',
            'init_randomness': 0.1,
            'diayn_reward': 0.1,
            "prob_perturb_action": 0.1,
            "perturb_action": 0.01,
            "alignment_reward": 80,
            "move_heading_reward": 80
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
        '''
        self.ob_shape = OrderedDict([(self.husky, 29), (self.box, 6)])
        '''
            Our husky model is differential drive, though it has 4 wheels.
            So, the action space should be 2
        '''
        self.action_space.decompose(OrderedDict([(self.husky, 2)]))

        if self._env_config["direction"] == 'right':
            self.dx, self.dy = 1, 0
        if self._env_config["direction"] == 'left':
            self.dx, self.dy = -1, 0
        if self._env_config["direction"] == 'forward':
            self.dx, self.dy = 0, 1
        if self._env_config["direction"] == 'backward':
            self.dx, self.dy = 0, -1

        

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

        self.do_simulation(a)
        #self.husky_simulation(a)
        pos_after = self._get_pos('husky_geom')
        box_after = self._get_pos('box_geom')
        box_quat_after = self._get_quat('box')

        husky_quat_after = self._get_quat('husky_robot')
        husky_forward_vector_after = right_vector_from_quat(husky_quat_after) 

        ob = self._get_obs()
        done = False
        alive_reward = 0
        ctrl_reward = self._ctrl_reward(a)


        # husky linear velocity
        '''
        husky_linear_vel = (pos_after[0] - pos_before[0]) * self.dx + \
            (pos_after[1] - pos_before[1]) * self.dy
        '''
    
        #husky_linear_vel = abs(pos_after[0] - pos_before[0]) + abs(pos_after[1] - pos_before[1])
        husky_linear_vel = l2_dist(pos_after, pos_before)
        husky_move_direction = forward_backward(self.data.qvel.ravel().copy())

        # husky angular velocity 
        husky_angular_vel = cos_dist(husky_forward_vector_before, husky_forward_vector_after)


        # travel distance
        offset = abs((pos_after[0] - self._husky_pos[0]) * self.dy) + \
            abs((pos_after[1] - self._husky_pos[1]) * self.dx)

        # box linear velocity
        '''
        box_linear_vel = (box_after[0] - box_before[0]) * self.dx + \
            (box_after[1] - box_before[1]) * self.dy
        '''
        #box_linear_vel = (box_after[0] - box_before[0]) + abs(box_after[1] - box_before[1])
        box_linear_vel = l2_dist(box_after, box_before)


        # box orientation on z-axis     # box forward vector after
        box_forward = right_vector_from_quat(box_quat_after)
        # box angular velocity
        box_angular_vel = cos_dist(self._box_forward, box_forward)


        # distance between box and husky
        box_dist = box_after - pos_after

        # reward components, for later use
        husky_linear_vel_reward = self._env_config["linear_vel_reward"] * husky_linear_vel
        husky_angular_vel_reward = self._env_config["angular_vel_reward"] * husky_angular_vel

        box_linear_vel_reward = self._env_config["box_linear_vel_reward"] * box_linear_vel
        box_angular_vel_reward = self._env_config["box_angular_vel_reward"] * (1- box_angular_vel)

        #offset_reward = self._env_config["offset_reward"] * (1 - min(2, offset))

        offset_reward = 0
        alive_reward = self._env_config["alive_reward"]

        # If robot's heading is parallel with object's heading?
        align_coeff = alignment_heading_difference(box_forward, husky_forward_vector_after)
        alignment_heading_reward = self._env_config["alignment_reward"] * align_coeff

        # if robot is moving toward the object?
        move_coeff = 0
        movement_heading_reward = 0


        # Failure check. Usually not.
        if not np.isfinite(self.data.qpos).all():
            done = True

        die_penalty = -self._env_config["die_penalty"] if done else 0
        #done = done or quat_dist < 0.9
        

        done = done or abs(box_dist[0]) > 5.5 or abs(box_dist[1]) > 5.5


        # Different reward functions for different primitive skills
        skill = self._env_config["direction"]

        if(skill == "right"):
            # encourage the robot to rotate correctly
            husky_angular_vel_reward = husky_angular_vel_reward * rotate_direction(husky_forward_vector_before, 
                                                                                   husky_forward_vector_after) 

            reward = ctrl_reward + alive_reward + die_penalty + husky_angular_vel_reward \
                + alignment_heading_reward 

        elif(skill == "left"):
            husky_angular_vel_reward = (-husky_angular_vel_reward) * rotate_direction(husky_forward_vector_before, 
                                                                                    husky_forward_vector_after)

            reward = ctrl_reward + alive_reward + die_penalty + husky_angular_vel_reward \
                + alignment_heading_reward

        elif(skill == "forward"):
            husky_linear_vel_reward = husky_linear_vel_reward * husky_move_direction

            move_coeff = movement_heading_difference(box_after, 
                                                     pos_after, 
                                                     husky_forward_vector_after, 
                                                     "forward")
            

            movement_heading_reward = self._env_config["move_heading_reward"] * move_coeff

            reward = ctrl_reward + alive_reward + die_penalty + offset_reward + husky_linear_vel_reward \
                + box_linear_vel_reward + box_angular_vel_reward + movement_heading_reward + husky_angular_vel_reward

        else:   # backward
            husky_linear_vel_reward = (-husky_linear_vel_reward) * husky_move_direction
             
            move_coeff = movement_heading_difference(box_after, 
                                                     pos_after,
                                                     husky_forward_vector_after,
                                                     "backward")

            movement_heading_reward = self._env_config["move_heading_reward"] * move_coeff

            reward = ctrl_reward + alive_reward + die_penalty + offset_reward + husky_linear_vel_reward \
                + box_linear_vel_reward + box_angular_vel_reward + movement_heading_reward + husky_angular_vel_reward
            

        '''
        self._reward = reward = vel_reward + \
            ctrl_reward + alive_reward + offset_reward + die_penalty + \
            quat_reward
        '''
        self._reward = reward

        info = {"Current Skill": skill,
                "Total Reward": reward,
                "reward: husky_linear": husky_linear_vel_reward,
                "reward: husky_angular": husky_angular_vel_reward,
                "reward: heading_alignment": alignment_heading_reward,
                "reward: movement_heading": movement_heading_reward,
                "husky_forward or backward": husky_move_direction,
                "husky_movement_direction_coeff": move_coeff,
                "husky_alignment_coeff": align_coeff,

                "reward: box_linear": box_linear_vel_reward,
                "reward: box_angular": box_angular_vel_reward,

                "reward_ctrl": ctrl_reward,
                "reward_alive": alive_reward,
                "penalty_die": die_penalty,
                "box_forward": box_forward,
                "husky_pos": pos_after,
                "box_pos": box_after,
                "box_ob": ob[self.box],
                "success": self._success}

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
        box_forward = self._get_forward_vector('box_geom')


        # You should be able to find the observation space in this scenario
        # Check _reset below to see what are in qpos
        # The same way could be also applied to qvel and qacc
        obs = OrderedDict([
            (self.husky, np.concatenate([qpos[2:11], qvel[:10], qacc[:10]])),
            (self.box, np.concatenate([box_pos - husky_pos, box_forward])),
            #('shared_pos', np.concatenate([qpos[2:7], qvel[:6], qacc[:6]])),
            #('lower_body', np.concatenate([qpos[7:15], qvel[6:14], qacc[6:14]])),
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

        return np.array([-2., 0., 0.2, 0., 0., 0., 0., 0., 0., 0., 0., 
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
        
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)

        # DON'T PUT HUSKY TOO HIGH!
        qpos[2] = 0.19
        qpos[3] = 0
        qpos[4] = 0
        qpos[5] = 0
        
        self.set_state(qpos, qvel)

        self._reset_husky_box()

        return self._get_obs()
    
    def _reset_husky_box(self):

        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()


        # Initialize box
        init_box_pos = np.asarray([0, 0, 0.8])
        init_box_quat = sample_quat()
        #init_box_quat = [1, 0, 0, 0]

        # reset the box, reset pos and quat respectively
        qpos[11:14] = init_box_pos
        qpos[14:18] = init_box_quat

        self._box_forward = right_vector_from_quat(init_box_quat)

        # Initialized Husky
        x = np.random.uniform(low=-3.0, high=-2.0)
        y = np.random.uniform(low=-1, high=1)


        direction = self._env_config["direction"]
        '''
        if direction == "right":
            pass
        elif direction == "left":
            x = -x
        elif direction == "forward":
            x, y = y, x
        elif direction == "backward":
            x, y = y, -x
        '''

        if direction == "forward":
            x, y = -x, y
        elif direction == "backward":
            pass
        elif direction == "left":
            x, y = -x, y
        elif direction == "right":
            pass


        qpos[0] = x
        qpos[1] = y
        self._husky_pos = (x, y)

        self.set_state(qpos, qvel)




        

    
    
    



