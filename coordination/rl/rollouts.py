from collections import defaultdict

import numpy as np
import torch
import cv2

from util.logger import logger


class Rollout(object):
    """ Rollout storage. """
    def __init__(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        batch = {}
        batch['ob'] = self._history['ob']
        batch['ac'] = self._history['ac']
        batch['meta_ac'] = self._history['meta_ac']
        batch['ac_before_activation'] = self._history['ac_before_activation']
        batch['done'] = self._history['done']
        batch['rew'] = self._history['rew']
        self._history = defaultdict(list)
        return batch


class MetaRollout(object):
    """ Rollout storage for meta policy. """
    def __init__(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        batch = {}
        batch['ob'] = self._history['meta_ob']
        batch['ac'] = self._history['meta_ac']
        batch['ac_before_activation'] = self._history['meta_ac_before_activation']
        batch['log_prob'] = self._history['meta_log_prob']
        batch['done'] = self._history['meta_done']
        batch['rew'] = self._history['meta_rew']
        self._history = defaultdict(list)
        return batch


class RolloutRunner(object):
    """ Rollout runner. """

    def __init__(self, config, env, meta_pi, pi):
        self._config = config
        self._env = env
        self._meta_pi = meta_pi
        self._pi = pi

    def run_episode(self, max_step=10000, is_train=True, record=False):
        """ Runs one episode and returns a rollout. """
        config = self._config
        device = config.device
        env = self._env
        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)
        acs = []

        done = False
        ep_len = 0
        ep_rew = 0
        ob = self._env.reset()
        if config.diayn and config.meta is None:
            sampled_z = pi._actors[0][0]._sample_z()
        else:
            sampled_z = None
        self._record_frames = []
        if record: self._store_frame()

        # buffer to save qpos
        saved_qpos = []

        # run rollout
        meta_ac = None
        while not done and ep_len < max_step:
            curr_meta_ac, meta_ac_before_activation, meta_log_prob = \
                meta_pi.act(ob, is_train=is_train)

            if meta_ac is None or (not config.fix_embedding):
                meta_ac = curr_meta_ac
            else:
                # logic for config.fix_embedding
                # check if curr_meta_ac has different skills selections from meta_ac
                # if so, then update meta_ac, else do not update
                assert config.diayn == True
                should_update_embedding = False
                for key in curr_meta_ac.keys():
                    if key.endswith('_diayn'):
                        continue
                    if curr_meta_ac[key] != meta_ac[key]:
                        should_update_embedding = True
                        break
                if should_update_embedding:
                    meta_ac = curr_meta_ac

            meta_rollout.add({
                'meta_ob': ob, 'meta_ac':  meta_ac,
                'meta_ac_before_activation': meta_ac_before_activation,
                'meta_log_prob': meta_log_prob,
            })

            meta_len = 0
            meta_rew = 0
            while not done and ep_len < max_step and meta_len < config.max_meta_len:
                ll_ob = ob.copy()

                if config.meta:
                    ac, ac_before_activation = pi.act(ll_ob, meta_ac, is_train=is_train)
                else:
                    if sampled_z is not None:
                        ll_ob.update(sampled_z)
                    ac, ac_before_activation = pi.act(ll_ob, is_train=is_train)

                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                saved_qpos.append(env.sim.get_state().qpos.copy())


                ob, reward, done, info = env.step(ac)

                # get discriminator output
                if sampled_z is not None:
                    diayn_rew = 0
                    for _agent in pi._actors:
                        for _actor in _agent:
                            actor_discriminator_loss = _actor.discriminator_loss()
                            if actor_discriminator_loss is not None:
                                diayn_rew += -actor_discriminator_loss.detach().cpu().item()
                    diayn_rew *= self._env._env_config['diayn_reward']
                    reward += diayn_rew
                    info['diayn_rew'] = diayn_rew

                rollout.add({'done': done, 'rew': reward})
                acs.append(ac)
                ep_len += 1
                ep_rew += reward
                meta_len += 1
                meta_rew += reward

                for key, value in info.items():
                    reward_info[key].append(value)
                if record:
                    frame_info = info.copy()
                    if sampled_z is not None:
                        frame_info['diayn_z'] = np.round(list(sampled_z.values())[0], 1).tolist()
                    if config.meta:
                        frame_info['meta_ac'] = []
                        for i, k in enumerate(meta_ac.keys()):
                            if not k.endswith('diayn'):
                                frame_info['meta_ac'].append(meta_pi.subdiv_skills[i][int(meta_ac[k])])
                            else:
                                frame_info[k] = meta_ac[k]

                    self._store_frame(frame_info)

            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})

        # last frame
        ll_ob = ob.copy()
        if sampled_z is not None:
            ll_ob.update(sampled_z)
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob})
        saved_qpos.append(env.sim.get_state().qpos.copy())

        ep_info = {'len': ep_len, 'rew': ep_rew}
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)
        ep_info['saved_qpos'] = saved_qpos

        return rollout.get(), meta_rollout.get(), ep_info, self._record_frames

    def _store_frame(self, info={}):
        """ Adds caption to the frame and adds it to a list of frames. """
        color = (200, 200, 200)

        text = "{:4} {}".format(self._env._episode_length,
                                self._env._episode_reward)
        frame = self._env.render('rgb_array') * 255.0
        fheight, fwidth = frame.shape[:2]
        frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

        if self._config.record_caption:
            font_size = 0.4
            thickness = 1
            offset = 12
            x, y = 5, fheight + 10
            cv2.putText(frame, text,
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (255, 255, 0), thickness, cv2.LINE_AA)
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = '{}: '.format(k)
                (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, thickness)

                cv2.putText(frame, key_text,
                            (x, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (66, 133, 244), thickness, cv2.LINE_AA)

                cv2.putText(frame, str(v),
                            (x + key_width, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (255, 255, 255), thickness, cv2.LINE_AA)

        self._record_frames.append(frame)

