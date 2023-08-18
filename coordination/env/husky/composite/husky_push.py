from collections import OrderedDict

import numpy as np

from env.husky.husky import HuskyEnv
from env.transform_utils import up_vector_from_quat, forward_vector_from_quat, \
    l2_dist, cos_dist, sample_quat, right_vector_from_quat


class HuskyPushEnv(HuskyEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_push'
        super().__init__('husky_push.xml', **kwargs)

        # Env info
        self.ob_shape = OrderedDict([("husky_1", 28), ("husky_2", 28),
                                     ("box_1", 6), ("box_2", 6),
                                     ("goal_1", 3), ("goal_2", 3)])
        
        
        self.action_space.decompose(OrderedDict([("husky_1", 2), ("husky_2", 2)]))

        # Env config
        self._env_config.update({
            'random_husky_pos': 0.01,
            'random_goal_pos': 0.01,
            #'random_goal_pos': 0.5,
            'dist_threshold': 0.3,
            'quat_threshold': 0.3,
            'husky_dist_reward': 10,

            'goal_dist_reward': 20,
            'goal1_dist_reward': 10,
            'goal2_dist_reward': 10,


            'H_to_H_dist_penalty': 200,
            'quat_reward': 50, # quat_dist usually between 0.95 ~ 1
            'alive_reward': 0.,
            'success_reward': 1000,
            'die_penalty': 10,
            'sparse_reward': 0,
            'init_randomness': 0.01,
            #'max_episode_steps': 400,
            'max_episode_steps': 500 ,
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
        goal_forward = right_vector_from_quat(goal_quat)
        box_forward = right_vector_from_quat(box_quat)
        box_forward_before = right_vector_from_quat(box_quat_before)

        # goal 1
        goal1_dist = l2_dist(goal1_pos, box1_pos)
        goal1_dist_before = l2_dist(goal1_pos, box1_pos_before)

        # goal 2
        goal2_dist = l2_dist(goal2_pos, box2_pos)
        goal2_dist_before = l2_dist(goal2_pos, box2_pos_before)

        # goal overall
        goal_dist = l2_dist(goal_pos, box_pos)
        goal_dist_before = l2_dist(goal_pos, box_pos_before)
        goal_quat = cos_dist(goal_forward, box_forward)
        goal_quat_before = cos_dist(goal_forward, box_forward_before)

        husky1_quat = self._get_quat('husky_robot_1')
        husky2_quat = self._get_quat('husky_robot_2')

        # husky 1 distance toward box_1, husky 2 distance toward box_2
        husky1_dist = l2_dist(husky1_pos_before[:2], box1_pos_before[:2]) - l2_dist(husky1_pos[:2], box1_pos[:2])
        husky2_dist = l2_dist(husky2_pos_before[:2], box2_pos_before[:2]) - l2_dist(husky2_pos[:2], box2_pos[:2])

        success_reward = 0

        # Even husky is pushing, the distance is still a concern. Better not to ignore them. However, the reward should be calculated in another way
        # if not push, then calculate the distance the robot has been moving forward to the target object, the more, the better
        # If not push, then it is Primitive Skill: Approach
        #husky1_dist_reward = self._env_config["husky_dist_reward"] * husky1_dist if not self._husky1_push else 0
        #husky2_dist_reward = self._env_config["husky_dist_reward"] * husky2_dist if not self._husky2_push else 0

        #husky1_dist_reward = self._env_config["husky_dist_reward"] * (1/(husky1_dist + 1))  # add 1 to avoid pure 0
        #husky2_dist_reward = self._env_config["husky_dist_reward"] * (1/(husky2_dist + 1))  # add 1 to avoid pure 0
        husky1_dist_reward = -self._env_config["husky_dist_reward"] * husky1_dist
        husky2_dist_reward = -self._env_config["husky_dist_reward"] * husky2_dist

        # Check the distance between box and goal
        goal1_dist_reward = -self._env_config["goal1_dist_reward"] * goal1_dist
        goal2_dist_reward = -self._env_config["goal2_dist_reward"] * goal2_dist
        goal_dist_reward = -self._env_config["goal_dist_reward"] * goal_dist

        '''
            Reward
        '''
        # Distance and Difference between two Huskys
        # Linear velocity differece
        husky1_linear_vel = l2_dist(husky1_pos_before, husky1_pos)
        husky2_lienar_vel = l2_dist(husky2_pos_before, husky2_pos)
        vel_diff_reward = abs(husky1_linear_vel - husky2_lienar_vel) * self._env_config["vel_diff_reward"]




        '''
            Penalty (not constraint)
        '''
        # Husky to husky distance penalty
        HH_dist_penalty = (1 / (l2_dist(husky1_pos[:2], husky2_pos[:2]) + 1)) * self._env_config["H_to_H_dist_penalty"]


        # Note: goal_quat is the cost_dist between goal and box 
        # NOTE: why "-"? doesn't make sense
        '''
        quat_reward = -self._env_config["quat_reward"] * (1 - goal_quat)
        '''
        quat_reward = self._env_config[("quat_reward")] * (1 - goal_quat)


        #quat_reward = self._env_config["quat_reward"] * (goal_quat - goal_quat_before) if goal_pos[0] >= box_pos[0] else 0
        alive_reward = self._env_config["alive_reward"]
        #ant1_height_reward = -self._env_config["height_reward"] * np.abs(ant1_height - 0.6)
        #ant2_height_reward = -self._env_config["height_reward"] * np.abs(ant2_height - 0.6)
        #ant1_upright_reward = self._env_config["upright_reward"] * ant1_upright
        #ant2_upright_reward = self._env_config["upright_reward"] * ant2_upright

        if goal1_dist < self._env_config["dist_threshold"] and \
                goal2_dist < self._env_config["dist_threshold"] and \
                goal_quat < self._env_config["quat_threshold"]:
            self._success_count += 1
            success_reward = 10
            if self._success_count >= 1:
                self._success = True
                success_reward = self._env_config["success_reward"]

        # fail
        #done = not np.isfinite(self.data.qpos).all() or \
        #    0.35 > ant1_height or ant1_height > 0.9 or \
        #    0.35 > ant2_height or ant2_height > 0.9
        #done = done or abs(ant1_pos[0]) > 7 or abs(ant2_pos[0]) > 7
        #done = done or abs(ant1_pos[1]) > 6 or abs(ant2_pos[1]) > 6
        die_penalty = -self._env_config["die_penalty"] if done else 0
        done = done or self._success

        if self._env_config['sparse_reward']:
            self._reward = reward = self._success == 1
        else:
            self._reward = reward = husky1_dist_reward + husky2_dist_reward + \
                goal_dist_reward + goal1_dist_reward + goal2_dist_reward + quat_reward + \
                alive_reward + success_reward + ctrl_reward + die_penalty
            


        info = {"success": self._success,
                "reward_husky1_dist": husky1_dist_reward,
                "reward_husky2_dist": husky2_dist_reward,
                "reward_goal_dist": goal_dist_reward,
                "reward_goal1_dist": goal1_dist_reward,
                "reward_goal2_dist": goal2_dist_reward,
                "reward_quat": quat_reward,
                "reward_alive": alive_reward,
                "reward_ctrl": ctrl_reward,
                "die_penalty": die_penalty,
                "reward_success": success_reward,
                "husky1_pos": husky1_pos,
                "husky2_pos": husky2_pos,
                "goal_quat": goal_quat,
                "goal_dist": goal_dist,
                "goal1_dist": goal1_dist,
                "goal2_dist": goal2_dist,
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

        obs = OrderedDict([
            #('ant_1_shared_pos', np.concatenate([qpos[:7], qvel[:6], qacc[:6]])),
            #('ant_1_lower_body', np.concatenate([qpos[7:15], qvel[6:14], qacc[6:14]])),
            #('ant_2_shared_pos', np.concatenate([qpos[15:22], qvel[14:22], qacc[14:22]])),
            #('ant_2_lower_body', np.concatenate([qpos[22:30], qvel[22:30], qacc[22:30]])),
            ('husky_1', np.concatenate([qpos[3:11], qvel[:10], qacc[:10]])),
            ('husky_2', np.concatenate([qpos[14:22], qvel[10:20], qacc[10:20]])),
            ('box_1', np.concatenate([box_pos1 - husky_pos1, box_forward1])),
            ('box_2', np.concatenate([box_pos2 - husky_pos2, box_forward2])),
            ('goal_1', goal_pos1 - box_pos1),
            ('goal_2', goal_pos2 - box_pos2),
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
        return np.array([-4., 1, 0.58, 0., 0., 0., 0., 0., 0, 0., 0.,
                         -4., -1, 0.58, 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0.58, 0., 0., 0., 0.])

    @property
    def _init_qvel(self):
        return np.zeros(self.model.nv)

    def _reset(self):
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)

        qpos[2] = 0.58
        qpos[13] = 0.58
        qpos[3:7] = [np.random.uniform(low=-1, high=1, size=1), 0, 0, np.random.uniform(low=-1, high=1, size=1)]
        qpos[14:18] = [np.random.uniform(low=-1, high=1, size=1), 0, 0, np.random.uniform(low=-1, high=1, size=1)]

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
        qpos[0:2] = [-4, 1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        qpos[11:13] = [-4, -1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        #qpos[0:2] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], 2]
        #qpos[15:17] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], -2]

        # Initialize goal
        #x = 4.5 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        x = 3 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        y = 0 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        z = 0.3
        goal_pos = np.asarray([x, y, z])
        #goal_quat = sample_quat()
        self._set_pos('goal', goal_pos)
        #self._set_quat('goal', goal_quat)

        self.set_state(qpos, qvel)

    def _render_callback(self):
        lookat = [0.5, 0, 0]
        cam_pos = lookat + np.array([-4.5, -12, 10])

        cam_id = self._camera_id
        self._set_camera_position(cam_id, cam_pos)
        self._set_camera_rotation(cam_id, lookat)

        self.sim.forward()

