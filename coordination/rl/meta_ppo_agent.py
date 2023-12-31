import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dataset import ReplayBuffer, RandomSampler
from rl.base_agent import BaseAgent
from rl.policies.mlp_actor_critic import MlpActor, MlpCritic
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, \
    obs2tensor, to_tensor
from env.action_spec import ActionSpec


class MetaPPOAgent(BaseAgent):
    """ Meta policy class. """

    def __init__(self, config, ob_space):
        super().__init__(config, ob_space)

        if config.meta is None:
            logger.warn('Creating a dummy meta policy.')
            return

        # parse body parts and skills
        if config.subdiv:
            # subdiv = ob1,ob2-ac1/ob3,ob4-ac2/...
            clusters = config.subdiv.split('/')
            clusters = [cluster.split('-')[1].split(',') for cluster in clusters]

            tmp_clusters = []
            for cluster in clusters:
                if cluster not in tmp_clusters:
                    tmp_clusters.append(cluster)
            clusters = tmp_clusters
            
        else:
            clusters = [ob_space.keys()]

        if config.subdiv_skills:
            subdiv_skills = config.subdiv_skills.split('/')
            subdiv_skills = [skills.split(',') for skills in subdiv_skills]
        else:
            subdiv_skills = [['primitive']] * len(clusters)
        self.subdiv_skills = subdiv_skills

        assert len(subdiv_skills) == len(clusters), \
            'subdiv_skills and clusters have different # subdivisions'

        if config.meta == 'hard':
            ac_space = ActionSpec(size=0)
            for cluster, skills in zip(clusters, subdiv_skills):
                ac_space.add(','.join(cluster), 'discrete', len(skills), 0, 1)
            self.ac_space = ac_space


        # ant_1,box_1_diayn
        # ant_2,box_2_diayn
        if config.diayn:
            ob_clusters = config.subdiv.split('/')
            ob_clusters = [cluster.split('-')[0].split(',') for cluster in ob_clusters]
            for cluster, skills in zip(ob_clusters, subdiv_skills):
                self.ac_space.add(','.join(cluster) + '_diayn', 'continuous', config.z_dim, 0, 1)

        # build up networks
        self._actor = MlpActor(config, ob_space, ac_space, tanh_policy=False)
        self._old_actor = MlpActor(config, ob_space, ac_space, tanh_policy=False)
        self._critic = MlpCritic(config, ob_space)
        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.lr_critic)

        sampler = RandomSampler()
        self._buffer = ReplayBuffer(['ob', 'ac', 'done', 'rew', 'ret', 'adv',
                                     'ac_before_activation', 'log_prob'],
                                    config.buffer_size,
                                    sampler.sample_func)

        if config.is_chef:
            logger.warn('Creating a meta PPO agent')
            logger.info('The actor has %d parameters', count_parameters(self._actor))
            logger.info('The critic has %d parameters', count_parameters(self._critic))

    def store_episode(self, rollouts):
        """ Stores @rollouts to replay buffer. """
        self._compute_gae(rollouts)
        self._buffer.store_episode(rollouts)

    def _compute_gae(self, rollouts):
        """ Computes GAE from @rollouts. """
        T = len(rollouts['done'])
        ob = rollouts['ob']
        ob = self.normalize(ob)
        ob = obs2tensor(ob, self._config.device)
        vpred = self._critic(ob).detach().cpu().numpy()[:,0]
        assert len(vpred) == T + 1

        done = rollouts['done']
        rew = rollouts['rew']
        adv = np.empty((T, ) , 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            delta = rew[t] + self._config.discount_factor * vpred[t + 1] * nonterminal - vpred[t]
            adv[t] = lastgaelam = delta + self._config.discount_factor * self._config.gae_lambda * nonterminal * lastgaelam

        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        if adv.std() == 0:
            rollouts['adv'] = (adv * 0).tolist()
        else:
            rollouts['adv'] = ((adv - adv.mean()) / adv.std()).tolist()
        rollouts['ret'] = ret.tolist()

    def state_dict(self):
        if self._config.meta is None:
            return {}

        return {
            'actor_state_dict': self._actor.state_dict(),
            'critic_state_dict': self._critic.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic_optim_state_dict': self._critic_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if self._config.meta is None:
            return

        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._critic.load_state_dict(ckpt['critic_state_dict'])
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self._critic_optim.load_state_dict(ckpt['critic_optim_state_dict'])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._old_actor.to(device)
        self._critic.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic)

    def train(self):
        self._copy_target_network(self._old_actor, self._actor)

        for _ in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions)

        self._buffer.clear()

        train_info.update({
            'actor_grad_norm': compute_gradient_norm(self._actor),
            'actor_weight_norm': compute_weight_norm(self._actor),
            'critic_grad_norm': compute_gradient_norm(self._critic),
            'critic_weight_norm': compute_weight_norm(self._critic),
        })
        return train_info

    def _update_network(self, transitions):
        info = {}

        # pre-process observations
        o = transitions['ob']
        o = self.normalize(o)

        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions['ac'])
        z = _to_tensor(transitions['ac_before_activation'])
        ret = _to_tensor(transitions['ret']).reshape(bs, 1)
        adv = _to_tensor(transitions['adv']).reshape(bs, 1)
        old_log_pi = _to_tensor(transitions['log_prob']).reshape(bs, 1)

        log_pi, ent = self._actor.act_log(o, z)

        if (log_pi - old_log_pi).max() > 20:
            print('(log_pi - old_log_pi) is too large', (log_pi - old_log_pi).max())
            import ipdb; ipdb.set_trace()

        # the actor loss
        entropy_loss = self._config.entropy_loss_coeff * ent.mean()
        ratio = torch.exp(torch.clamp(log_pi - old_log_pi, -20, 20))
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self._config.clip_param,
                            1.0 + self._config.clip_param) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        if not np.isfinite(ratio.cpu().detach()).all() or not np.isfinite(adv.cpu().detach()).all():
            import ipdb; ipdb.set_trace()

        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        discriminator_loss = self._actor.discriminator_loss()
        if discriminator_loss is not None:
            actor_loss += discriminator_loss * self._config.discriminator_loss_weight
            info['discriminator_loss'] = discriminator_loss.cpu().item()

        # the q loss
        value_pred = self._critic(o)
        value_loss = self._config.value_loss_coeff * (ret - value_pred).pow(2).mean()

        info['value_target'] = ret.mean().cpu().item()
        info['value_predicted'] = value_pred.mean().cpu().item()
        info['value_loss'] = value_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self._actor)
        self._actor_optim.step()

        # update the critic
        self._critic_optim.zero_grad()
        value_loss.backward()
        sync_grads(self._critic)
        self._critic_optim.step()

        # include info from policy
        info.update(self._actor.info)

        return mpi_average(info)

    def act(self, ob, is_train=True):
        """
        Returns a set of actions and the actors' activations given an observation @ob.
        """
        if self._config.meta:
            ob = self.normalize(ob)
            return self._actor.act(ob, is_train, return_log_prob=True)
        else:
            return [0], None, None

