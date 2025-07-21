import itertools
import math
import random

import gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from gym import spaces
from PIL import Image
from io import BytesIO

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GridNavi(gym.Env):
    def __init__(self, num_cells=5, num_steps=15, starting_state=(0.0, 0.0), ring=False, hall_with_door_every=None, 
                 new_state_r=None, stuck=False, distance_reward=False, show_goal_at_start=False):
        if starting_state is None:
            assert not ring and hall_with_door_every is None, "Random start location not supported for ring or hall"
        super(GridNavi, self).__init__()

        self.seed()
        self.num_cells = num_cells
        self.num_states = num_cells ** 2 # should be  num_cells * 2 if hallway, but this messes up visualization 
        self.new_state_r = new_state_r # reward for looking at a potential goal
        self.distance_reward = distance_reward
        self.show_goal_at_start = show_goal_at_start

        self._max_episode_steps = num_steps
        self.step_count = 0

        self.hall_with_door_every = hall_with_door_every # whether to train in a hallway with doors (potential goal) every x steps
        self.stuck = stuck # Whether the agent gets "stuck" behind the door they enter

        self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(2,)) if hall_with_door_every is None \
            else spaces.Box(low=np.array([0,0]), high=np.array([self.num_cells-1,1]))
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.task_dim = 2
        self.belief_dim = num_cells ** 2 

        # possible starting states
        self.starting_state = starting_state

        # define possible goal locations
        if starting_state is not None:
            sx, sy = starting_state
            sx, sy = int(sx), int(sy)
        if ring: #goals on perimeter
            assert hall_with_door_every is None, "Cannot have ring and hallway"
            self.possible_goals = set()
            for offset in range(num_cells):
                self.possible_goals.add((0, offset)) #left side
                self.possible_goals.add((num_cells-1, offset)) #right side
                self.possible_goals.add((offset, num_cells-1)) #top
                self.possible_goals.add((offset, 0)) #bottom
            if (sx, sy) in self.possible_goals:
                self.possible_goals.remove((sx, sy))
            self.possible_goals = list(self.possible_goals)
        elif hall_with_door_every is not None:
            assert starting_state[1] == 0, "Must start in hall if hall_with_door_every is not None"
            self.possible_goals = []
            for offset in range(hall_with_door_every, num_cells, hall_with_door_every): # door every x steps, but not at start
                self.possible_goals.append((offset, 1))
        else:
            self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
            if starting_state is not None:
                # goals can be anywhere except on possible starting states and immediately around it
                for x_offset in [-1, 0, 1]:
                    for y_offset in [-1, 0, 1]:
                        pos = (sx+x_offset, sy+y_offset)
                        if pos in self.possible_goals:
                            self.possible_goals.remove(pos)

        # keep a list of places the agent has not checked yet
        self.goals_unchecked = set(self.possible_goals)

        self.task_dim = 2
        self.num_tasks = num_cells ** 2 # Note: this should really be len(self.possible_goals), but that complicates task_to_id()

        # reset the environment state
        self._env_state = np.array(random.choice(self.possible_goals)) if starting_state is None else np.array(self.starting_state)
        # reset the goal
        self._goal = self.reset_task()
        # reset the belief
        self._belief_state = self._reset_belief()

    def reset_task(self, task=None):
        if task is None:
            self._goal = np.array(random.choice(self.possible_goals))
        else:
            self._goal = np.array(task)
        self._reset_belief()
        self.goals_unchecked = set(self.possible_goals)
        return self._goal

    def _reset_belief(self):
        self._belief_state = np.zeros((self.num_cells ** 2))
        for pg in self.possible_goals:
            idx = self.task_to_id(np.array(pg))
            self._belief_state[idx] = 1.0 / len(self.possible_goals)
        return self._belief_state

    def update_belief(self, state, action):

        on_goal = state[0] == self._goal[0] and state[1] == self._goal[1]

        # hint
        if action == 5 or on_goal:
            possible_goals = self.possible_goals.copy()
            possible_goals.remove(tuple(self._goal))
            wrong_hint = possible_goals[random.choice(range(len(possible_goals)))]
            self._belief_state *= 0
            self._belief_state[self.task_to_id(self._goal)] = 0.5
            self._belief_state[self.task_to_id(wrong_hint)] = 0.5
        else:
            self._belief_state[self.task_to_id(state)] = 0
            self._belief_state = np.ceil(self._belief_state)
            self._belief_state /= sum(self._belief_state)

        return self._belief_state

    def get_task(self):
        return self._goal.copy()

    def get_belief(self):
        return self._belief_state.copy()

    def reset(self):
        self.step_count = 0
        self._env_state = np.array(random.choice(self.possible_goals)) if self.starting_state is None else np.array(self.starting_state)
        return self._env_state.copy()

    def state_transition(self, action):
        """
        Moving the agent between states
        """
        if self.hall_with_door_every is not None and self._env_state[1] == 1:
            if self.stuck:
                return self._env_state # Cannot go backward once a door is entered!
            elif action != 3:
                return self._env_state # Can only go down once a door is entered!

        if action == 1:  # up
            # can only go up if not in hallway or there is door above agent
            if self.hall_with_door_every is None:
                self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
            elif (int(self._env_state[0]), int(self._env_state[1]+1)) in self.possible_goals:
                self._env_state[1] = self._env_state[1]+1
        elif action == 2:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 4:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])

        return self._env_state

    def step(self, action):

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        done = False

        # perform state transition
        state = self.state_transition(action)

        # keep track of potential goals checked
        if self.new_state_r is not None:
            checked_potential_goal = False
            sx, sy = state
            sx, sy = int(sx), int(sy)
            if (sx, sy) in self.goals_unchecked:
                checked_potential_goal = True
                self.goals_unchecked.remove((sx, sy))

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        # compute reward
        if self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]:
            reward = 1.0
        elif self.new_state_r is not None and checked_potential_goal:
            reward = self.new_state_r
        elif self.distance_reward:
            reward = (-abs(self._env_state[0] - self._goal[0]) - abs(self._env_state[1] - self._goal[1]))/self.num_cells
        else:
            reward = -0.1

        # update ground-truth belief
        self.update_belief(self._env_state, action)

        task = self.get_task()
        task_id = self.task_to_id(task)
        info = {'task': task,
                'task_id': task_id,
                'belief': self.get_belief()}

        if self.show_goal_at_start:
            assert self.starting_state is not None, "Start state must be specified to show goal there"
            if self._env_state[0] == self.starting_state[0] and self._env_state[1] == self.starting_state[1]:
                state = self._goal[:] # show goal as obs

        return state.copy(), reward, done, info

    def task_to_id(self, goals):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells))
        if isinstance(goals, list) or isinstance(goals, tuple):
            goals = np.array(goals)
        if isinstance(goals, np.ndarray):
            goals = torch.from_numpy(goals)
        goals = goals.long()

        if goals.dim() == 1:
            goals = goals.unsqueeze(0)

        goal_shape = goals.shape
        if len(goal_shape) > 2:
            goals = goals.reshape(-1, goals.shape[-1])

        classes = mat[goals[:, 0], goals[:, 1]]
        classes = classes.reshape(goal_shape[:-1])

        return classes

    def id_to_task(self, classes):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells)).numpy()
        goals = np.zeros((len(classes), 2))
        classes = classes.numpy()
        for i in range(len(classes)):
            pos = np.where(classes[i] == mat)
            goals[i, 0] = float(pos[0][0])
            goals[i, 1] = float(pos[1][0])
        goals = torch.from_numpy(goals).to(device).float()
        return goals

    def goal_to_onehot_id(self, pos):
        cl = self.task_to_id(pos)
        if cl.dim() == 1:
            cl = cl.view(-1, 1)
        nb_digits = self.num_cells ** 2
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(pos.shape[0], nb_digits).to(device)
        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, cl, 1)
        return y_onehot

    def onehot_id_to_goal(self, pos):
        if isinstance(pos, list):
            pos = [self.id_to_task(p.argmax(dim=1)) for p in pos]
        else:
            pos = self.id_to_task(pos.argmax(dim=1))
        return pos

    def render(self):
        num_cells_x = int(self.observation_space.high[0] + 1)
        num_cells_y = int(self.observation_space.high[1] + 1)

        state = self._env_state.astype(int)
        goal = self._goal.astype(int)

        plt.figure(figsize=(5, 5))
        plt.axis('off')
        
        # Draw grid
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                rec = Rectangle((i, j), 1, 1, facecolor='w', alpha=0.5, edgecolor='k')
                plt.gca().add_patch(rec)
        
        # Plot state and goal with larger markers and high z-order
        plt.plot(state[0] + .5, num_cells_y - state[1] - 1 + .5, 'bo', markersize=10, zorder=5)
        plt.plot(goal[0] + .5, num_cells_y - goal[1] - 1 + .5, 'gx', markersize=15, zorder=5)  # Changed to green for visibility

        # Make it look nice
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, num_cells_x])
        plt.ylim([num_cells_y, 0])  # Invert the y-axis limits

        # Save plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()

        # Open the image from the buffer and convert to RGB
        buffer.seek(0)
        image = Image.open(buffer)
        image_rgb = np.array(image.convert('RGB'))

        return image_rgb


    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            image_folder=None,
                            **kwargs
                            ):
        """
        Visualises the behaviour of the policy, together with the latent state and belief.
        The environment passed to this method should be a SubProcVec or DummyVecEnv, not the raw env!
        """

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0]

        # --- initialise things we want to keep track of ---

        episode_all_obs = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        episode_goals = []
        if args.pass_belief_to_policy and (encoder is None):
            episode_beliefs = [[] for _ in range(num_episodes)]
        else:
            episode_beliefs = None

        if encoder is not None:
            # keep track of latent spaces
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

        # --- roll out policy ---

        env.reset_task()
        [state, belief, task] = utl.reset_env(env, args)
        start_obs = state.clone()

        for episode_idx in range(args.max_rollouts_per_task):

            curr_goal = env.get_task()
            curr_rollout_rew = []
            curr_rollout_goal = []

            if encoder is not None:

                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            episode_all_obs[episode_idx].append(start_obs.clone())
            if args.pass_belief_to_policy and (encoder is None):
                episode_beliefs[episode_idx].append(belief)

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    prev_obs = start_obs.clone()
                else:
                    prev_obs = state.clone()
                episode_prev_obs[episode_idx].append(prev_obs)

                # act
                _, action, _ = utl.select_action(args=args,
                                                 policy=policy,
                                                 state=state.view(-1),
                                                 belief=belief,
                                                 task=task,
                                                 deterministic=True,
                                                 latent_sample=curr_latent_sample.view(-1) if (curr_latent_sample is not None) else None,
                                                 latent_mean=curr_latent_mean.view(-1) if (curr_latent_mean is not None) else None,
                                                 latent_logvar=curr_latent_logvar.view(-1) if (curr_latent_logvar is not None) else None,
                                                 )

                # observe reward and next obs
                [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(env, action, args)

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.float().to(device),
                        state,
                        rew_raw.reshape((1, 1)).float().to(device),
                        prev_obs,
                        hidden_state,
                        return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_all_obs[episode_idx].append(state.clone())
                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())

                curr_rollout_rew.append(rew_raw.clone())
                curr_rollout_goal.append(env.get_task().copy())

                if args.pass_belief_to_policy and (encoder is None):
                    episode_beliefs[episode_idx].append(belief)

                if infos[0]['done_mdp'] and not done:
                    start_obs = infos[0]['start_state']
                    start_obs = torch.from_numpy(start_obs).float().reshape((1, -1)).to(device)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)
            episode_goals.append(curr_goal)

        # clean up

        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot behaviour & visualise belief in env

        rew_pred_means, rew_pred_vars = plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
                                                episode_latent_means, episode_latent_logvars,
                                                image_folder, iter_idx, episode_beliefs)

        if reward_decoder:
            plot_rew_reconstruction(env, rew_pred_means, rew_pred_vars, image_folder, iter_idx)

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns


def plot_rew_reconstruction(env,
                            rew_pred_means,
                            rew_pred_vars,
                            image_folder,
                            iter_idx,
                            ):
    """
    Note that env might need to be a wrapped env!
    """

    num_rollouts = len(rew_pred_means)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    test_rew_mus = torch.cat(rew_pred_means).cpu().detach().numpy()
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus, '.-', alpha=0.5)
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_mus.max() - test_rew_mus.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_mus.min() - 0.05 * span, test_rew_mus.max() + 0.05 * span], 'k--',
                 alpha=0.5)
    plt.title('output - mean')

    plt.subplot(1, 3, 2)
    test_rew_vars = torch.cat(rew_pred_vars).cpu().detach().numpy()
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars, '.-', alpha=0.5)
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_vars.max() - test_rew_vars.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_vars.min() - 0.05 * span, test_rew_vars.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('output - variance')

    plt.subplot(1, 3, 3)
    rew_pred_entropy = -(test_rew_vars * np.log(test_rew_vars)).sum(axis=1)
    plt.plot(range(len(test_rew_vars)), rew_pred_entropy, 'r.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = rew_pred_entropy.max() - rew_pred_entropy.min()
        plt.plot([tj + 0.5, tj + 0.5], [rew_pred_entropy.min() - 0.05 * span, rew_pred_entropy.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('Reward prediction entropy')

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_rew_decoder'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()


def plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
            episode_latent_means, episode_latent_logvars, image_folder, iter_idx, episode_beliefs):
    """
    Plot behaviour and belief.
    """

    # only_plot_last_step = env._max_episode_steps > 30 
    only_plot_last_step = False  # Always plot all steps for gridworld

    plt.figure(figsize=(1.5 * env._max_episode_steps, 1.5 * args.max_rollouts_per_task))

    num_episodes = len(episode_all_obs)
    num_steps = len(episode_all_obs[0])

    rew_pred_means = [[] for _ in range(num_episodes)]
    rew_pred_vars = [[] for _ in range(num_episodes)]

    # loop through the experiences
    for episode_idx in range(num_episodes):
        step_range = [num_steps-1] if only_plot_last_step else range(num_steps) # long horizons just plot at end
        for step_idx in step_range:

            curr_obs = episode_all_obs[episode_idx][:step_idx + 1]
            curr_goal = episode_goals[episode_idx]

            if episode_latent_means is not None:
                curr_means = episode_latent_means[episode_idx][:step_idx + 1]
                curr_logvars = episode_latent_logvars[episode_idx][:step_idx + 1]

            # choose correct subplot
            plt.subplot(args.max_rollouts_per_task,
                        math.ceil(env._max_episode_steps) + 1,
                        1 + episode_idx * (1 + math.ceil(env._max_episode_steps)) + step_idx)

            # plot the behaviour
            plot_behaviour(env, curr_obs, curr_goal)

            if reward_decoder is not None:
                # visualise belief in env
                rm, rv = compute_beliefs(env,
                                         args,
                                         reward_decoder,
                                         curr_means[-1],
                                         curr_logvars[-1],
                                         curr_goal)
                rew_pred_means[episode_idx].append(rm)
                rew_pred_vars[episode_idx].append(rv)
                plot_belief(env, rm, args)
            elif episode_beliefs is not None:
                curr_beliefs = episode_beliefs[episode_idx][step_idx]
                plot_belief(env, curr_beliefs, args)
            else:
                rew_pred_means = rew_pred_vars = None

            if episode_idx == 0:
                plt.title('t = {}'.format(step_idx))

            if step_idx == 0:
                plt.ylabel('Episode {}'.format(episode_idx + 1))

    if reward_decoder is not None:
        rew_pred_means = [torch.stack(r) for r in rew_pred_means]
        rew_pred_vars = [torch.stack(r) for r in rew_pred_vars]

    # save figure that shows policy behaviour
    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()

    return rew_pred_means, rew_pred_vars


def plot_behaviour(env, observations, goal):
    num_cells_x = int(env.observation_space.high[0] + 1)
    num_cells_y = int(env.observation_space.high[1] + 1)

    # draw grid
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='none', alpha=0.5,
                            edgecolor='k')
            plt.gca().add_patch(rec)

    # shift obs and goal by half a stepsize
    if isinstance(observations, tuple) or isinstance(observations, list):
        observations = torch.cat(observations)
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    # visualise behaviour, current position, goal
    plt.plot(observations[:, 0], observations[:, 1], 'b-')
    plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')

    # make it look nice
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, num_cells_x])
    plt.ylim([0, num_cells_y])


def compute_beliefs(env, args, reward_decoder, latent_mean, latent_logvar, goal):
    num_cells_x = int(env.observation_space.high[0] + 1)
    # num_cells_y = int(env.observation_space.high[1] + 1)
    unwrapped_env = env.venv.unwrapped.envs[0]

    if not args.disable_stochasticity_in_latent:
        # take several samples fromt he latent distribution
        samples = utl.sample_gaussian(latent_mean.view(-1), latent_logvar.view(-1), 100)
    else:
        samples = torch.cat((latent_mean.view(-1), latent_logvar.view(-1))).unsqueeze(0)

    # compute reward predictions for those
    if reward_decoder.multi_head:
        rew_pred = reward_decoder(samples, None)
        if args.rew_pred_type == 'bernoulli':
            rew_pred = torch.sigmoid(rew_pred)
        elif args.rew_pred_type == 'categorical':
            rew_pred = torch.softmax(rew_pred, 1)
        rew_pred_means = torch.mean(rew_pred, dim=0)  # .reshape((1, -1))
        rew_pred_vars = torch.var(rew_pred, dim=0)  # .reshape((1, -1))
    else:
        tsm = []
        tsv = []
        for st in range(num_cells_x*num_cells_x):
            task_id = unwrapped_env.id_to_task(torch.tensor([st]))
            curr_state = unwrapped_env.goal_to_onehot_id(task_id).expand((samples.shape[0], 2))
            if unwrapped_env.oracle:
                if isinstance(goal, np.ndarray):
                    goal = torch.from_numpy(goal)
                curr_state = torch.cat((curr_state, goal.repeat(curr_state.shape[0], 1).float()), dim=1)
            rew_pred = reward_decoder(samples, curr_state)
            if args.rew_pred_type == 'bernoulli':
                rew_pred = torch.sigmoid(rew_pred)
            tsm.append(torch.mean(rew_pred))
            tsv.append(torch.var(rew_pred))
        rew_pred_means = torch.stack(tsm).reshape((1, -1))
        rew_pred_vars = torch.stack(tsv).reshape((1, -1))
    # rew_pred_means = rew_pred_means[-1][0]

    return rew_pred_means, rew_pred_vars


def plot_belief(env, beliefs, args):
    """
    Plot the belief by taking 100 samples from the latent space and plotting the average predicted reward per cell.
    """

    num_cells_x = int(env.observation_space.high[0] + 1)
    num_cells_y = int(env.observation_space.high[1] + 1)
    unwrapped_env = env.venv.unwrapped.envs[0]

    # draw probabilities for each grid cell
    alphas = []
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            pos_i = i
            pos_j = j
            idx = unwrapped_env.task_to_id(torch.tensor([[pos_i, pos_j]]))
            alpha = beliefs[idx]
            alphas.append(alpha.item())
    alphas = np.array(alphas)
    # cut off values (this only happens if we don't use sigmoid/softmax)
    alphas[alphas < 0] = 0
    alphas[alphas > 1] = 1
    # alphas = (np.array(alphas)-min(alphas)) / (max(alphas) - min(alphas))
    count = 0
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='r', alpha=alphas[count],
                            edgecolor='k')
            plt.gca().add_patch(rec)
            count += 1
