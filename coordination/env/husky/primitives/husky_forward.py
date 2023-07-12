from collections import OrderedDict

import numpy as np

from env.ant.ant import AntEnv
from env.transform_utils import up_vector_from_quat, forward_vector_from_quat, \
    l2_dist, cos_dist, right_vector_from_quat, sample_quat


class AntForwardEnv(AntEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_forward'
        super().__init__('husky_forward.xml', **kwargs)

        # Env config
        self._env_config.update({
            'vel_reward': 50,
            'box_vel_reward': 20,
            'offset_reward': 1,
            'height_reward': 0.5,
            'upright_reward': 0.5,
            'alive_reward': 0.,
            'quat_reward': 0,
            'die_penalty': 10,
            'max_episode_steps': 200,
            'husky': 1,
            'direction': 'right',
            'init_randomness': 0.05,
            'diayn_reward': 0.1,
            "prob_perturb_action": 0.1,
            "perturb_action": 0.01,
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        # Env info
        self.husky = "husky_%d" % self._env_config['husky']
        self.box = "box_%d" % self._env_config['husky']
        self.ob_shape = OrderedDict([(self.husky, 41), (self.box, 6)])
        self.action_space.decompose(OrderedDict([(self.husky, 8)]))
        if self._env_config["direction"] == 'right':
            self.dx, self.dy = 1, 0
        if self._env_config["direction"] == 'left':
            self.dx, self.dy = -1, 0
        if self._env_config["direction"] == 'up':
            self.dx, self.dy = 0, 1
        if self._env_config["direction"] == 'down':
            self.dx, self.dy = 0, -1

    def _step(self, a):
        pos_before = self._get_pos('husky_geom')
        box_before = self._get_pos('box_geom')
        a = self._perturb_action(a)
        self.do_simulation(a)
        pos_after = self._get_pos('husky_geom')
        box_after = self._get_pos('box_geom')

        ob = self._get_obs()
        done = False
        vel_reward = 0
        height_reward = 0
        alive_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        height = self.data.qpos[2]

        # velocity?
        vel = (pos_after[0] - pos_before[0]) * self.dx + \
            (pos_after[1] - pos_before[1]) * self.dy

        #offset = abs(pos_after[0] * self.dy) - abs(pos_before[0] * self.dy) + \
        #    abs(pos_after[1] * self.dx) - abs(pos_before[1] * self.dx)
        offset = abs((pos_after[0] - self._ant_pos[0]) * self.dy) + \
            abs((pos_after[1] - self._ant_pos[1]) * self.dx)

        box_vel = 0
        box_vel = (box_after[0] - box_before[0]) * self.dx + \
            (box_after[1] - box_before[1]) * self.dy

        box_forward = self._get_forward_vector('box_geom')
        quat_dist = cos_dist(self._box_forward, box_forward)

        box_dist = box_after - pos_after

        husky_quat = self._get_quat('husky_robot')
        up = up_vector_from_quat(husky_quat)
        upright = cos_dist(up, np.array([0, 0, 1]))

        # reward
        vel_reward = self._env_config["vel_reward"] * vel
        box_vel_reward = self._env_config["box_vel_reward"] * box_vel
        offset_reward = self._env_config["offset_reward"] * (1 - min(2, offset))
        height_reward = -self._env_config["height_reward"] * np.abs(height - 0.6)
        alive_reward = self._env_config["alive_reward"]
        quat_reward = self._env_config["quat_reward"] * (quat_dist - 1)
        upright_reward = self._env_config["upright_reward"] * upright

        # fail
        done = not np.isfinite(self.data.qpos).all() or \
            0.35 > height or height > 0.9
        die_penalty = -self._env_config["die_penalty"] if done else 0
        #done = done or quat_dist < 0.9
        done = done or abs(box_dist[0]) > 5.5 or abs(box_dist[1]) > 5.5

        self._reward = reward = vel_reward + height_reward + \
            ctrl_reward + alive_reward + offset_reward + die_penalty + \
            quat_reward + upright_reward

        info = {"reward_vel": vel_reward,
                "reward_box_vel": box_vel_reward,
                "reward_offset": offset_reward,
                "reward_ctrl": ctrl_reward,
                "reward_height": height_reward,
                "reward_alive": alive_reward,
                "reward_quat": quat_reward,
                "reward_upright": upright_reward,
                "penalty_die": die_penalty,
                "box_forward": box_forward,
                "ant_pos": pos_after,
                "box_ob": ob[self.box],
                "success": self._success}

        return ob, reward, done, info

