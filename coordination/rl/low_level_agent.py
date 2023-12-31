import os
from collections import OrderedDict

import numpy as np
import torch

from rl.sac_agent import SACAgent
from rl.normalizer import Normalizer
from util.logger import logger
from util.pytorch import to_tensor, get_ckpt_path
from env.action_spec import ActionSpec


class LowLevelAgent(SACAgent):
    """
    Low level agent that includes skill sets for each agent, their
    execution procedure given observation and skill selections from
    meta-policy, and their training (for single-skill-per-agent cases
    only).
    """

    def __init__(self, config, ob_space, ac_space, actor, critic):
        super().__init__(config, ob_space, ac_space, actor, critic)

    def _log_creation(self):
        """ Logs the structure of low-level policies. """
        if self._config.is_chef:
            logger.info('Creating a low-level agent')
            for cluster, skills in zip(self._clusters, self._subdiv_skills):
                logger.warn('Part {} has skills {}'.format(cluster, skills))

    def _build_actor(self, actor):
        config = self._config

        # parse body parts and skills
        if config.subdiv:
            # subdiv: 'ob1,ob2-ac1/ob3,ob4-ac2/...'
            clusters = config.subdiv.split('/')
            clusters = [
                (cluster.split('-')[0].split(','), cluster.split('-')[1].split(',')) for cluster in clusters
            ]
        else:
            clusters = [(ob_space.keys(), ac_space.shape.keys())]


        if config.subdiv_skills:
            subdiv_skills = config.subdiv_skills.split('/')
            subdiv_skills = [
                skills.split(',') for skills in subdiv_skills
            ]
        else:
            subdiv_skills = [['primitive']] * len(clusters)


        assert len(subdiv_skills) == len(clusters), \
            'subdiv_skills and clusters have different # subdivisions'

        self._clusters = clusters
        self._subdiv_skills = subdiv_skills

        self._actors = []
        self._ob_norms = []

        # load networks
        for cluster, skills in zip(self._clusters, self._subdiv_skills):
            ob_space = OrderedDict([(k, self._ob_space[k]) for k in cluster[0]])


            if self._config.diayn:
                ob_space[','.join(cluster[0]) + '_diayn'] = self._config.z_dim



            ac_decomposition = OrderedDict([(k, self._ac_space.shape[k]) for k in cluster[1]])
            ac_size = sum(self._ac_space.shape[k] for k in cluster[1])
            ac_space = ActionSpec(ac_size, -1, 1)
            ac_space.decompose(ac_decomposition)

            skill_actors = []
            skill_ob_norms = []
            for skill in skills:
                skill_actor = actor(config, ob_space, ac_space, config.tanh_policy)
                skill_ob_norm = Normalizer(ob_space,
                                           default_clip_range=config.clip_range,
                                           clip_obs=config.clip_obs)
                
                if self._config.meta_update_target == 'HL':
                    path = os.path.join(config.subdiv_skill_dir, skill)
                    ckpt_path, ckpt_num = get_ckpt_path(path, None)
                    logger.warn('Load skill checkpoint (%s) from (%s)', skill, ckpt_path)
                    ckpt = torch.load(ckpt_path)

                    if type(ckpt['agent']['actor_state_dict']) == OrderedDict:
                        # backward compatibility to older checkpoints
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'])
                    else:
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'][0][0])
                    skill_ob_norm.load_state_dict(ckpt['agent']['ob_norm_state_dict'])

                skill_actor.to(config.device)
                skill_actors.append(skill_actor)
                skill_ob_norms.append(skill_ob_norm)

            self._actors.append(skill_actors)
            self._ob_norms.append(skill_ob_norms)

    def act(self, ob, meta_ac, is_train=True):
        """
        Returns action and the actor's activation given an observation @ob and meta action @meta_ac for rollout.
        """
        ac = OrderedDict()
        activation = OrderedDict()
        if self._config.meta == 'hard':
            for i, skill_idx in enumerate(meta_ac.values()):
                if [k for k in meta_ac.keys()][i].endswith('_diayn'):
                    # skip diayn outputs from meta-policy
                    continue
                skill_idx = skill_idx[0]
                ob_ = ob.copy()
                if self._config.diayn:
                    z_name = self._actors[i][skill_idx].z_name
                    ob_[z_name] = meta_ac[self._actors[i][skill_idx].z_name]
                ob_ = self._ob_norms[i][skill_idx].normalize(ob_)
                ob_ = to_tensor(ob_, self._config.device)
                if self._config.meta_update_target == 'HL':
                    ac_, activation_ = self._actors[i][skill_idx].act(ob_, False)
                else:
                    ac_, activation_ = self._actors[i][skill_idx].act(ob_, is_train)
                ac.update(ac_)
                activation.update(activation_)

        return ac, activation

    def act_log(self, ob, meta_ac=None):
        """
        Returns action and the actor's activation given an observation @ob and meta action @meta_ac for updating networks.
        Note: only usable for SAC agents.
        """
        ob_detached = { k: v.detach().cpu().numpy() for k, v in ob.items() }

        ac = OrderedDict()
        log_probs = []
        meta_ac_keys = [k for k in meta_ac.keys() if (not k.endswith('_diayn'))]
        for i, key in enumerate(meta_ac_keys):
            #skill_idx = meta_ac[key]
            skill_idx = 0

            ob_ = ob_detached.copy()
            if self._config.diayn:
                z_name = self._actors[i][skill_idx].z_name
                ob_[z_name] = meta_ac[z_name].detach().cpu().numpy()
            ob_ = self._ob_norms[i][skill_idx].normalize(ob_)
            ob_ = to_tensor(ob_, self._config.device)
            ac_, log_probs_ = self._actors[i][skill_idx].act_log(ob_)
            ac.update(ac_)
            log_probs.append(log_probs_)

        try:
            log_probs = torch.cat(log_probs, -1).sum(-1, keepdim=True)
        except Exception:
            import pdb; pdb.set_trace()

        return ac, log_probs

    def sync_networks(self):
        if self._config.meta_update_target == 'LL' or \
           self._config.meta_update_target == 'both':
            super().sync_networks()
        else:
            pass
