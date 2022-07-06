from copy import deepcopy
import itertools
from envs.maTTenv import agent_models
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
from gym.spaces import Box, Discrete
import time, os, random, pdb
from utils.logSpinUp import EpochLogger
import algos.maTT.core as core
from algos.maTT.replay_buffer import ReplayBufferSet as ReplayBuffer

import argparse
import os
import random
import time
from distutils.util import strtobool
from tqdm import tqdm

import gym
from stable_baselines.common.cmd_util import make_vec_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        key = list(envs.single_observation_space.keys())[0]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space[key].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space[key].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def decentralized_ppo(envs, args, run_name):
    """
    env_fn: 
        lambda: env
    """
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = make_vec_env(env_fn, n_envs=args.num_envs, vec_env_cls=gym.vector.SyncVectorEnv)
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent_network = PPO(envs)
    optimizer = optim.Adam(agent_network.parameters(), lr=args.learning_rate, eps=1e-5)
    # agent = Agent(envs).to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    """
    A single obs, actions, logprob, done, value

    obs:
        OrderedDict()
            keys: [agent-ids]
            values: (num_envs, nb_targets, observation)
                observation --> [[d, alpha, ddot, alphadot, logdet(Sigma), observed]]
    action:
        dict()
            keys: [agent-ids]
            values: torch.Tensor() of torch.Size[num_envs]
    logprob:
        dict()
            keys: [agent-ids]
            values: torch.Tensor() of torch.Size[num_envs]
    dones:
        dict()
            keys: [agent-ids]
            values: torch.Tensor() of torch.Size[num_envs]
    values:
        dict()
            keys: [agent-ids]
            values: torch.Tensor() of torch.Size[num_envs]
    """
    obs = torch.zeros((args.num_steps, args.num_envs, args.nb_agents) + envs.single_observation_space["agent-0"].shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, args.nb_agents) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, args.nb_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, args.nb_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            #for agent_id in optimizers.keys():
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            for i, agent_id in enumerate(next_obs.keys()):
                obs[step, :, i] = torch.from_numpy(next_obs[agent_id])
            dones[step] = next_done # dones[step] = next_done
            # ALGO LOGIC: action logic
            action_dict = [{} for _ in range(args.num_envs)]
            with torch.no_grad():
                for i, agent_id in enumerate(next_obs.keys()):
                    action, logprob, _, value = agent_network.get_action_and_value(next_obs[agent_id])
                    for j in range(action.shape[0]): # num_envs
                        action_dict[j][agent_id] = action[j].item()
                    actions[step, :, i] = action.to(device)
                    logprobs[step, :, i] = logprob.to(device)
                    values[step, :, i] = value.view(-1).to(device)
            
            if args.render:
                envs.envs[0].render()
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action_dict)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(done).to(device)
            if "episode" in info.keys():
                assert torch.sum(next_done).item() == info["episode"].shape[0]
                for env in range(args.num_envs):
                    print(f"global_step={global_step}, episodic_return={info['episode'][env]['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"][env]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"][env]["l"], global_step)
                break
        
        
        # bootstrap value if not done
        advantages = torch.zeros((args.num_steps, args.num_envs, args.nb_agents))
        returns = torch.zeros((args.num_steps, args.num_envs, args.nb_agents))
        for i, agent_id in enumerate(next_obs.keys()): # 1st problem: is how to deal with multiple agents?
        # how to bootstrap with multiple agents and a global reward
            with torch.no_grad():
                next_value = agent_network.get_value(next_obs[agent_id]).reshape(1, -1)
                if args.gae:
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1, :, i]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t, :, i]
                        advantages[t, :, i] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns[:, :, i] = advantages[:, :, i] + values[:, :, i]
                else:
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1, :, i]
                        returns[t, :, i] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages[:, :, i] = returns[:, :, i] - values[:, :, i]
            
            # flatten the batch
            b_obs = obs[:, :, i].reshape((-1,) + envs.single_observation_space["agent-0"].shape)
            b_logprobs = logprobs[:, :, i].reshape(-1)
            b_actions = actions[:, :, i].reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages[:, :, i].reshape(-1)
            b_returns = returns[:, :, i].reshape(-1)
            b_values = values[:, :, i].reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    # logprobs are being compared. make sure the log probs are referencing the same targets
                    _, newlogprob, entropy, newvalue = agent_network.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds]) # b_obs[mb_inds] --> (64, num_targets, observation.dim)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent_network.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    if args.record:
        print('finished')
        envs.envs[0].moviewriter.finish()
    envs.close()
    writer.close()