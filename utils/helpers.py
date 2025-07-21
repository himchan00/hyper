import os
import gym
import pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def make_env(args, mode='train', train_task_override=None, **kwargs):
    env_id = args.env_name

    # NEW ENV: METAWORLD
    if env_id.startswith('metaworld'):

        if args.mw_version == 1:
            from environments.metaworld import metaworld
        elif args.mw_version == 2:
            from environments.metaworld_v2 import metaworld

        env_type = 'metaworld'

        # --- ML1 ---
        # import the right meta-world-environment
        if env_id == 'metaworld_ml1':
            env_name = f'{args.ml1_type}-v{args.mw_version}'
            mworld = metaworld.ML1(env_name)  # Construct the benchmark, sampling tasks
            # set up train/test env
            if mode == 'train':
                env = mworld.train_classes[env_name]()
                if train_task_override is not None:
                    env.reset_task = lambda: env.set_task(random.choice(train_task_override))
                else:
                    env.reset_task = lambda: env.set_task(random.choice(mworld.train_tasks))
            elif mode == 'test':
                env = mworld.test_classes[env_name]()
                env.reset_task = lambda: env.set_task(random.choice(mworld.test_tasks))

        # --- ML10 ---
        elif env_id == 'metaworld_ml10':
            ml10 = metaworld.ML10()
            # if mode == 'train':
            #     n_envs = 10
            # elif mode == 'test':
            #     n_envs = 5
            # else:
            #     raise ValueError

            # n_tasks = n_envs * 1  # Leo: This ensures 1 env of each is sampled.
            from environments.garage.experiment.task_sampler import MetaWorldTaskSampler # Can't do this at top since it breaks MuJoCo131 needed for Walker
            task_sampler = MetaWorldTaskSampler(ml10,
                                                mode,  # train or test
                                                wrapper=None,
                                                # lambda env, _: normalize(env),  # TODO: not sure if we should use this
                                                add_env_onehot=False)
            # envs = [env_up() for env_up in task_sampler.sample(n_tasks)]
            from environments.mw_wrapper import MetaWorldMultiEnvWrapper # Can't do this at top since it breaks MuJoCo131 needed for Walker
            env = MetaWorldMultiEnvWrapper(task_sampler,
                                           n_tasks_train=10,
                                           n_tasks_test=5, # needed to make one-hot ids
                                           mode='vanilla',
                                           train=(mode=='train'))
        else:
            raise ValueError
        env._max_episode_steps = env.max_path_length
    elif env_id.startswith('T-') or env_id.startswith('MC-'):
        env = gym.make(env_id, **kwargs)
        env_type = "Maze"
    # OTHERWISE WE ASSUME ITS A GYM ENV
    else:
        env_type = 'gym'
        if args is not None and args.env_name == 'RoomNavi-v0':
            env = gym.make(env_id,
                           num_cells=args.num_cells,
                           corridor_len=args.corridor_len,
                           num_steps=args.horizon,
                           **kwargs)
        if args is not None and args.env_name == 'TreasureHunt-v0':
            env = gym.make(env_id,
                           max_episode_steps=args.max_episode_steps,
                           mountain_height=args.mountain_height,
                           treasure_reward=args.treasure_reward,
                           timestep_penalty=args.timestep_penalty,
                           **kwargs)
        elif args is not None and args.env_name == 'AntGoalSparse-v0':
            env = gym.make(env_id,
                           level=args.level,
                           **kwargs)
        else:
            env = gym.make(env_id, **kwargs)

    return env, env_type


def reset_env(env, args, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if indices is not None:
        assert not isinstance(indices[0], bool)
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)

    belief = torch.from_numpy(env.get_belief()).to(device) if args.pass_belief_to_policy else None
    decode_task_in_rlloss = args.decode_task and args.rlloss_through_encoder
    task = torch.from_numpy(env.get_task()).to(device) if args.pass_task_to_policy or decode_task_in_rlloss else None
        
    return state, belief, task


def env_step(env, action, args):

    next_obs, reward, done, infos = env.step(action.detach())

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    decode_task_in_rlloss = args.decode_task and args.rlloss_through_encoder
    task = torch.from_numpy(env.get_task()).to(device) if args.pass_task_to_policy or decode_task_in_rlloss else None

    return [next_obs, belief, task], reward, done, infos


def select_action(args,
                  policy,
                  deterministic,
                  state=None,
                  belief=None,
                  task=None,
                  latent_sample=None, latent_mean=None, latent_logvar=None,
                  return_latent = False,
                  training=False):
    """ Select action using the policy. """
    latent = get_latent_for_policy(args=args, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
    action = policy.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic, training=training)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action, action_log_prob = action
    else:
        value = None
        action_log_prob = None
    action = action.to(device)
    if return_latent:
        return value, action, action_log_prob, latent
    return value, action, action_log_prob


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):

    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None

    if args.add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if args.sample_embeddings:
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1:
        latent = latent.squeeze(0)

    return latent


def update_encoding(encoder, next_obs, action, reward, prev_obs, done, hidden_state):
    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        hidden_state = encoder.reset_hidden(hidden_state, done)

    with torch.no_grad():
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder(actions=action.float(),
                                                                          states=next_obs,
                                                                          rewards=reward,
                                                                          prev_states=prev_obs,
                                                                          hidden_state=hidden_state,
                                                                          return_prior=False)

    # TODO: move the sampling out of the encoder!

    return latent_sample, latent_mean, latent_logvar, hidden_state


def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'If you want identical results, use --num_processes 1 and --deterministic_execution True '
              '(only recommended for debugging).')


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def recompute_embeddings(
        policy_storage,
        encoder,
        sample,
        update_idx,
        detach_every
):
    # get the prior
    latent_sample = [policy_storage.latent_samples[0].detach().clone()]
    latent_mean = [policy_storage.latent_mean[0].detach().clone()]
    latent_logvar = [policy_storage.latent_logvar[0].detach().clone()]

    latent_sample[0].requires_grad = True
    latent_mean[0].requires_grad = True
    latent_logvar[0].requires_grad = True

    # loop through experience and update hidden state
    # (we need to loop because we sometimes need to reset the hidden state)
    h = policy_storage.hidden_states[0].detach()
    for i in range(policy_storage.actions.shape[0]):
        # reset hidden state of the GRU when we reset the task
        h = encoder.reset_hidden(h, policy_storage.done[i + 1])
        # detach the hidden state every detach_every steps
        if detach_every and (i%detach_every==0) and i!=0:
            h = h.detach()

        ts, tm, tl, h = encoder(policy_storage.actions.float()[i:i + 1],
                                policy_storage.next_state[i:i + 1],
                                policy_storage.rewards_raw[i:i + 1],
                                policy_storage.prev_state[i:i + 1],
                                h,
                                sample=sample,
                                return_prior=False,
                                )

        latent_sample.append(ts)
        latent_mean.append(tm)
        latent_logvar.append(tl)

    if update_idx == 0:
        try:
            assert (torch.cat(policy_storage.latent_mean) - torch.cat(latent_mean)).sum() == 0
            assert (torch.cat(policy_storage.latent_logvar) - torch.cat(latent_logvar)).sum() == 0
        except AssertionError:
            warnings.warn('You are not recomputing the embeddings correctly!')
            import pdb
            pdb.set_trace()

    policy_storage.latent_samples = latent_sample
    policy_storage.latent_mean = latent_mean
    policy_storage.latent_logvar = latent_logvar


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def sample_gaussian(mu, logvar, num=None):
    std = torch.exp(0.5 * logvar)
    if num is not None:
        std = std.repeat(num, 1)
        mu = mu.repeat(num, 1)
    eps = torch.randn_like(std)
    return mu + std * eps


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

def int_or_none(value):
    """Convert a string value to int or None."""
    return None if value.lower()=="none" else int(value)

def float_or_none(value):
    """Convert a string value to float or None."""
    return None if value.lower()=="none" else float(value)

def str_or_none(value):
    """Convert a string value to str or None."""
    return None if (value is None or value.lower()=="none") else str(value)

def get_task_dim(args):
    env, _ = make_env(args)
    try:
        return env.task_dim
    except AttributeError:
        return env.unwrapped.task_dim
    # Note: to be safe you may want to wrap in DummyVecEnv or some wrapper that handles attribute unwrapping instead
    return env.task_dim

def get_num_tasks(args):
    env, _ = make_env(args)
    try:
        num_tasks = env.num_tasks
    except AttributeError:
        print("NO TASK: env does not have attr env.num_tasks. This may cause issues for discrete task inference or one-hot encodings.")
        num_tasks = None
    return num_tasks


def clip(value, low, high):
    """Imitates `{np,tf}.clip`.

    `torch.clamp` doesn't support tensor valued low/high so this provides the
    clip functionality.

    TODO(hartikainen): The broadcasting hasn't been extensively tested yet,
        but works for the regular cases where
        `value.shape == low.shape == high.shape` or when `{low,high}.shape == ()`.
    """
    low, high = torch.tensor(low), torch.tensor(high)

    assert torch.all(low <= high), (low, high)

    clipped_value = torch.max(torch.min(value, high), low)
    return clipped_value
