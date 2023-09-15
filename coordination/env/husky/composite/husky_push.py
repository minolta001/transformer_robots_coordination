from collections import OrderedDict

import numpy as np

from env.husky.husky import HuskyEnv
from env.transform_utils import up_vector_from_quat, Y_vector_from_quat, \
    l2_dist, cos_dist, sample_quat, X_vector_from_quat, alignment_heading_difference, \
    Y_vector_overlapping, movement_heading_difference


class HuskyPushEnv(HuskyEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_push'
        super().__init__('husky_push.xml', **kwargs)

        # Env info
        self.ob_shape = OrderedDict([("husky_1", 31), ("husky_2", 31),
                                     ("box_1", 6), ("box_2", 6),
                                     ("goal_1", 3), ("goal_2", 3),
                                     ("relative_info_1", 2), ("relative_info_2", 2)])
        
        
        self.action_space.decompose(OrderedDict([("husky_1", 2), ("husky_2", 2)]))

        # Env config
        self._env_config.update({
            'random_husky_pos': 0.01,
            'random_goal_pos': 0.01,
            #'random_goal_pos': 0.5,
            'dist_threshold': 0.1,
            'loose_dist_threshold': 0.4,
            'goal_box_cos_dist_coeff_threshold': 0.95,

            'dist_reward': 10,
            'alignment_reward': 30,
            'goal_dist_reward': 30,
            'goal1_dist_reward': 10,
            'goal2_dist_reward': 10,
            'move_heading_reward': 10,
            'box_linear_vel_reward': 1000,
            'success_reward': 200,
            'bonus_reward': 20,

            'quat_reward': 200, # quat_dist usually between 0.95 ~ 1
            'alive_reward': 10,
            'die_penalty': 50,
            'sparse_reward': 0,
            'init_randomness': 0.01,
            #'max_episode_steps': 400,
            'max_episode_steps': 1000,
        }) 
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        self._husky1_push = False
        self._husky2_push = False
        self._success_count = 0

    def _step(self, a):
        husky1_pos_before = self._get_pos('husky_1_geom')
        husky2_pos_before = self._get_pos('husky_2_geom')

        box1_pos_before = self._get_pos('box_geom1')
        box2_pos_before = self._get_pos('box_geom2')
        box_pos_before = self._get_pos('box')
        box_quat_before = self._get_quat('box')

        self.do_simulation(a)

        husky1_pos = self._get_pos('husky_1_geom')
        husky2_pos = self._get_pos('husky_2_geom')
        box1_pos = self._get_pos('box_geom1')
        box2_pos = self._get_pos('box_geom2')
        box_pos = self._get_pos('box')
        box_quat = self._get_quat('box')
        goal1_pos = self._get_pos('goal_geom1')
        goal2_pos = self._get_pos('goal_geom2')
        goal_pos = self._get_pos('goal')
        goal_quat = self._get_quat('goal')

        ob = self._get_obs()
        done = False
        ctrl_reward = self._ctrl_reward(a)

        '''
        goal_forward = forward_vector_from_quat(goal_quat)
        box_forward = forward_vector_from_quat(box_quat)
        box_forward_before = forward_vector_from_quat(box_quat_before)
        '''

        #goal_forward_before = right_vector_from_quat(goal_quat_before)    
        goal_forward = X_vector_from_quat(goal_quat)
        box_forward = X_vector_from_quat(box_quat)
        box_forward_before = X_vector_from_quat(box_quat_before)

        # goal 1
        goal1_dist = l2_dist(goal1_pos, box1_pos)
        goal1_dist_before = l2_dist(goal1_pos, box1_pos_before)

        # goal 2
        goal2_dist = l2_dist(goal2_pos, box2_pos)
        goal2_dist_before = l2_dist(goal2_pos, box2_pos_before)

        husky1_quat = self._get_quat('husky_robot_1')
        husky2_quat = self._get_quat('husky_robot_2')


        '''
            Reward & Penalty
            PART 1, 2, 3: control the formation of huskys
            PART 4, 5: distance between husky and box, box and goal
            PART 6: move heading of huskys
        ''' 
        # PART 1: Forward parallel between two Huskys (checking forward vector)
        husky1_forward_vec = X_vector_from_quat(husky1_quat)
        husky2_forward_vec = X_vector_from_quat(husky2_quat)
        huskys_forward_align_coeff, dir = alignment_heading_difference(husky1_forward_vec, husky2_forward_vec)
        huskys_forward_align_reward = huskys_forward_align_coeff * self._env_config["alignment_reward"]

        # PART 2: Right vector overlapping between two Huskys (checking if two vectors are on the same line and same direction)
        # Actually, if Part 2 is gauranteed, then Part 1 is gauranteedgoal_box_cos_dist
        husky1_right_vec = Y_vector_from_quat(husky1_quat)
        husky2_right_vec = Y_vector_from_quat(husky2_quat)
        huskys_right_align_coeff = Y_vector_overlapping(husky1_right_vec, husky2_right_vec, husky1_pos, husky2_pos)
        huskys_right_align_reward = huskys_right_align_coeff * self._env_config["alignment_reward"]
        
        # PART 3: Distance between two Huskys (to avoid Collision)
        suggested_dist = l2_dist(box1_pos, box2_pos)
        huskys_dist = l2_dist(husky1_pos, husky2_pos)
        huskys_dist_reward = -abs(suggested_dist - huskys_dist) * self._env_config["dist_reward"]

        # PART 4: Linear distance between one husky and one box
        husky1_box_dist = l2_dist(husky1_pos, box1_pos)
        husky2_box_dist = l2_dist(husky2_pos, box2_pos)
        #husky1_box_dist_reward = -husky1_box_dist * self._env_config["dist_reward"]
        #husky2_box_dist_reward = -husky2_box_dist * self._env_config["dist_reward"]
        husky1_box_dist_reward = (5 - husky1_box_dist) * self._env_config["dist_reward"]
        husky2_box_dist_reward = (5 - husky2_box_dist) * self._env_config["dist_reward"]
        huskys_box_dist_reward = husky1_box_dist_reward + husky2_box_dist_reward
        
        # PART 5: Linear distance between box and goal
        goal1_box_dist = l2_dist(goal1_pos, box1_pos)
        goal2_box_dist = l2_dist(goal2_pos, box2_pos)
        #goal1_box_dist_reward = -goal1_box_dist * self._env_config["dist_reward"]
        #goal2_box_dist_reward = -goal2_box_dist * self._env_config["dist_reward"]
        goal1_box_dist_reward = (5 - goal1_box_dist) * self._env_config["goal_dist_reward"]
        goal2_box_dist_reward = (5 - goal2_box_dist) * self._env_config["goal_dist_reward"]
        goal_box_dist_reward = goal1_box_dist_reward + goal2_box_dist_reward



        # PART 6: Movement heading of husky to box
        husky1_move_coeff = movement_heading_difference(box1_pos, husky1_pos, husky1_forward_vec,)
        husky2_move_coeff = movement_heading_difference(box2_pos, husky2_pos, husky2_forward_vec)
        husky1_move_heading_reward = husky1_move_coeff * self._env_config["move_heading_reward"]
        husky2_move_heading_reward = husky2_move_coeff * self._env_config["move_heading_reward"]
        huskys_move_heading_reward = husky1_move_heading_reward + husky2_move_heading_reward

        # PART 7: Box velocity
        box1_linear_vel =  l2_dist(box1_pos, box1_pos_before)
        box2_linear_vel =  l2_dist(box2_pos, box2_pos_before)
        box1_linear_vel_reward = box1_linear_vel * self._env_config["box_linear_vel_reward"]
        box2_linear_vel_reward = box2_linear_vel * self._env_config["box_linear_vel_reward"]
        box_linear_vel_reward = box1_linear_vel_reward + box2_linear_vel_reward


        # PART 8: Cos distance between box and goal
        goal_box_cos_dist_coeff = 1 - cos_dist(goal_forward, box_forward) 
        goal_box_cos_dist_coeff = abs(goal_box_cos_dist_coeff - 0.5) / 0.5      # the larger goal_box_cos_dist_coeff, the better
        goal_box_cos_dist_reward = goal_box_cos_dist_coeff * self._env_config["quat_reward"]

        # PART 9 (Not sure side-effect): huskys velocity control   
        # We want huskys can stop/slow down when the box is close to the goal
        husky1_linear_vel = l2_dist(husky1_pos, husky1_pos_before)
        husky2_linear_vel = l2_dist(husky2_pos, husky2_pos_before)
        huskys_linear_vel_reward = 0 
        if (goal1_box_dist < 0.3) and husky1_linear_vel <= 0.005:
            huskys_linear_vel_reward += 1000
        if (goal2_box_dist < 0.3) and husky2_linear_vel <= 0.005:
            huskys_linear_vel_reward += 1000


        # PART 10: Linear distance between goal and huskys
        goal1_husky1_dist = l2_dist(goal1_pos, husky1_pos)
        goal2_husky2_dist = l2_dist(goal2_pos, husky2_pos)
        goal1_husky1_dist_reward = (5 - goal1_husky1_dist) * self._env_config["goal_dist_reward"]
        goal2_husky2_dist_reward = (5 - goal2_husky2_dist) * self._env_config["goal_dist_reward"]
        goal_huskys_dist_reward = goal1_husky1_dist_reward + goal2_husky2_dist_reward




        # Note: goal_quat is the cost_dist between goal and box 
        # NOTE: why "-"? doesn't make sense
        '''
        quat_reward = -self._env_config["quat_reward"] * (1 - goal_quat)
        '''
        quat_reward = self._env_config[("quat_reward")] * (1 - goal_quat)

        reward = 0
        alive_reward = self._env_config["alive_reward"] 
        reward += alive_reward


        '''
            Bonus
        '''
        if husky1_box_dist < 1.5 and husky2_box_dist < 1.5:
            reward += self._env_config['bonus_reward']
        if husky1_move_coeff > 0.92 and husky2_move_coeff > 0.92:
            reward += self._env_config['bonus_reward']
        if huskys_right_align_coeff > 0.92:
            reward += self._env_config['bonus_reward']
        if goal_box_cos_dist_coeff > 0.9:
            reward += (3 * self._env_config['bonus_reward'])

        '''
            Failure Check
        '''
        if huskys_dist < 1 or huskys_dist > 3.0:   # huskys are too close or too far away 
            done = True
        if husky1_box_dist > 6.0 or husky2_box_dist > 6.0: # husky is too far away from box
            done = True
        die_penalty = -self._env_config["die_penalty"] if done else 0

        # give some bonus if pass all failure check
        if done != True:
            reward += self._env_config["alive_reward"]

        '''
            Success Check
        '''
        success_reward = 0
        if (goal1_dist <= self._env_config["dist_threshold"] and \
            goal2_dist <= self._env_config["dist_threshold"]) or \
            (goal1_dist <= self._env_config["loose_dist_threshold"] and \
             goal2_dist <= self._env_config["loose_dist_threshold"] and \
             goal_box_cos_dist_coeff >= self._env_config["goal_box_cos_dist_coeff_threshold"]):
                # if both goal1 and goal2 suffice, then overall goal should suffice
                #and goal_quat < self._env_config["quat_threshold"]:
            self._success_count += 1
            success_reward = 10
            reward += success_reward
            if self._success_count >= 1:
                self._success = True
                success_reward = self._env_config["success_reward"]
                reward += success_reward
                done = True


        if self._env_config['sparse_reward']:
            self._reward = reward = self._success == 1
        else:

            reward = reward \
                    + huskys_forward_align_reward \
                    + huskys_dist_reward \
                    + huskys_box_dist_reward \
                    + goal_box_dist_reward \
                    + goal_huskys_dist_reward \
                    + huskys_move_heading_reward \
                    + goal_box_cos_dist_reward \
                    + huskys_linear_vel_reward
                    #+ huskys_right_align_reward \
                    #+ box_linear_vel_reward
            self._reward = reward


        info = {"success": self._success,
                "Total reward": reward,
                "reward: huskys forward align reward": huskys_forward_align_reward,
                "reward: huskys right align reward": huskys_right_align_reward,
                "reward: husky-to-husky dist reward": huskys_dist_reward,
                "reward: husky-to-box dist reward": huskys_box_dist_reward,
                "reward: goal-box dist reward": goal_box_dist_reward,
                "reward: move heading reward": huskys_move_heading_reward,
                "reward: goal_huskys_dist_reward": goal_huskys_dist_reward,
                #"reward: box velocity reward": box_linear_vel_reward,
                "reward: goal-to-box cos dist reward": goal_box_cos_dist_reward,
                "reward: huskys_vel_control_reward": huskys_linear_vel_reward,
                "coeff: huskys forward align coeff": huskys_forward_align_coeff,
                "coeff: huskys right align coeff": huskys_right_align_coeff,
                "coeff: husky1 move heading coeff": husky1_move_coeff,
                "coeff: husky2 move heading coeff": husky2_move_coeff,
                "die_penalty": die_penalty,
                "reward_success": success_reward,
                "huskys_dist": huskys_dist,
                #"husky1_pos": husky1_pos,
                #"husky2_pos": husky2_pos,
                "husky1_linear_vel": husky1_linear_vel,
                "husky2_lienar_vel": husky2_linear_vel,
                "goal1_husky1_dist": goal1_husky1_dist,
                "goal2_husky2_dist": goal2_husky2_dist,
                "box_goal_cos_dist": goal_box_cos_dist_coeff,
                "box1_ob": ob['box_1'],
                "box2_ob": ob['box_2'],
                "goal1_ob": ob['goal_1'],
                "goal2_ob": ob['goal_2'],
                }

        return ob, reward, done, info

    def _get_obs(self):
        # husky
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        husky_pos1 = self._get_pos('husky_robot_1')
        husky_pos2 = self._get_pos('husky_robot_2')

        # box
        box_pos1 = self._get_pos('box_geom1')
        box_pos2 = self._get_pos('box_geom2')

        #box_quat1 = self._get_quat('box')
        #box_quat2 = self._get_quat('box')

        box_forward1 = self._get_right_vector('box_geom1')
        box_forward2 = self._get_right_vector('box_geom2')

        # goal
        goal_pos1 = self._get_pos('goal_geom1')
        goal_pos2 = self._get_pos('goal_geom2')


        husky1_forward_vec = X_vector_from_quat(self._get_quat("husky_robot_1"))
        husky2_forward_vec = X_vector_from_quat(self._get_quat("husky_robot_2"))
        husky1_move_coeff = movement_heading_difference(box_pos1, husky_pos1, husky1_forward_vec, "forward")
        husky2_move_coeff = movement_heading_difference(box_pos2, husky_pos2, husky2_forward_vec, "forward")
        husky1_align_coeff, direction1 = alignment_heading_difference(box_forward1, husky1_forward_vec)
        husky2_align_coeff, direction2 = alignment_heading_difference(box_forward2, husky2_forward_vec)



        obs = OrderedDict([
            #('ant_1_shared_pos', np.concatenate([qpos[:7], qvel[:6], qacc[:6]])),
            #('ant_1_lower_body', np.concatenate([qpos[7:15], qvel[6:14], qacc[6:14]])),
            #('ant_2_shared_pos', np.concatenate([qpos[15:22], qvel[14:22], qacc[14:22]])),
            #('ant_2_lower_body', np.concatenate([qpos[22:30], qvel[22:30], qacc[22:30]])),
            ('husky_1', np.concatenate([qpos[3:11], qvel[:10], qacc[:10], husky1_forward_vec])),
            ('husky_2', np.concatenate([qpos[14:22], qvel[10:20], qacc[10:20], husky2_forward_vec])),
            ('box_1', np.concatenate([box_pos1 - husky_pos1, box_forward1])),
            ('box_2', np.concatenate([box_pos2 - husky_pos2, box_forward2])),
            ('goal_1', goal_pos1 - box_pos1),
            ('goal_2', goal_pos2 - box_pos2),
            ('relative_info_1', [husky1_move_coeff, husky1_align_coeff]),
            ('relative_info_2', [husky2_move_coeff, husky2_align_coeff])
        ])

        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    @property
    def _init_qpos(self):
        # 3 for (x, y, z), 4 for (x, y, z, w), and 2 for each leg
        # If I am correct, one line is for ant1, one line for ant2, the last line is for the box
        '''
        return np.array([0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0., 0., 0.8, 1., 0., 0., 0.])
        '''
    
        '''
            The structure of qpose of a single Husky and Box can be found in husky_forward.py _rest()
            
            The first and second 11 elements are for Huskys
            
            The last 7 are for box 
        '''
        return np.array([-2., 1, 0.58, 0., 0., 0., 0., 0., 0, 0., 0.,
                         -2., -1, 0.58, 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0.58, 0., 0., 0., 0.])

    @property
    def _init_qvel(self):
        return np.zeros(self.model.nv)

    def _reset(self):
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)

        qpos[2] = 0.2
        qpos[13] = 0.2
        qpos[3:7] = [0, 0, 0, 0]
        qpos[14:18] = [0, 0, 0, 0]

        self.set_state(qpos, qvel)

        self._reset_husky_box()

        self._husky1_push = False
        self._husky2_push = False
        self._success_count = 0

        return self._get_obs()

    def _reset_husky_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # Initialize box
        init_box_pos = np.asarray([0, 0, 0.3])
        #init_box_quat = sample_quat(low=-np.pi/32, high=np.pi/32)
        init_box_quat = sample_quat(low=0, high=0)
        qpos[22:25] = init_box_pos
        qpos[25:29] = init_box_quat

        # Initialize husky
        #qpos[0:2] = [-4, 2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        #qpos[15:17] = [-4, -2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        qpos[0:2] = [-2, 1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        qpos[11:13] = [-2, -1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        #qpos[0:2] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], 2]
        #qpos[15:17] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], -2]

        # Initialize goal
        #x = 4.5 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        x = 3 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        y = 0 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        z = 0.3

        goal_pos = np.asarray([x, y, z])
        #goal_quat = sample_quat(low=-np.pi/9, high=np.pi/9)
        goal_quat = sample_quat(low=-np.pi/6, high=np.pi/6)
        self._set_pos('goal', goal_pos)
        self._set_quat('goal', goal_quat)

        self.set_state(qpos, qvel)

    def _render_callback(self):
        lookat = [0.5, 0, 0]
        cam_pos = lookat + np.array([-4.5, -12, 10])

        cam_id = self._camera_id
        self._set_camera_position(cam_id, cam_pos)
        self._set_camera_rotation(cam_id, lookat)

        self.sim.forward()

