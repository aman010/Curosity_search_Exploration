#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 17:00:48 2025

@author: qb
"""


"""
Two-Agent Synchronous PPO for SC2 with Observation Augmentation
- Intrinsic curiosity bonus
- Macro-strategic risk (R_MS)
- Observation augmentation pipeline (NumPy + OpenCV)
- Live plotting of rewards, curiosity, risk

Save as `two_agent_ppo_sc2_augmented.py` and run: python two_agent_ppo_sc2_augmented.py
Requirements: pysc2 (StarCraft II Python API), OpenCV (cv2), torch, matplotlib, numpy
"""

import os
import time
import csv
import random
from collections import namedtuple
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import cv2
import multiprocessing as mp
from queue import Queue
from threading import Thread
import matplotlib.pyplot as plt

# ---------- USER CONFIG ----------
MAP_NAME = "AbyssalReefLE"
MODELS_DIR = f"models/{int(time.time())}/"
os.makedirs(MODELS_DIR, exist_ok=True)

NUM_AGENTS = 2
OBS_SHAPE = (224, 224, 3)
N_ACTIONS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOTAL_UPDATES = 200
STEPS_PER_UPDATE = 256
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LR = 2.5e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

INTRINSIC_WEIGHT = 0.8
ALPHA = 0.1
BETA = 0.25
MAX_GAME_VALUE = 40000.0
LAMBDA = 1.0

LOG_INTERVAL = 1
SAVE_INTERVAL = 50

# ---------- Observation augmentation parameters ----------
AUG_PROB = 0.8
AUG_CROP_PCT = 0.08
AUG_SHIFT_PCT = 0.08
AUG_BRIGHTNESS = 0.12
AUG_CONTRAST = 0.12
AUG_NOISE_STD = 6.0
AUG_CUTOUT_PCT = 0.12
AUG_FLIP_PROB = 0.5

# ---------- SC2 imports (must be available in your environment) ----------
from sc2 import maps
from sc2.data import Race
from sc2.main import run_game
from sc2.player import Bot
# Your VRBot: ensure signature VRBot(action_in=..., result_out=..., player_index=...)
from VoidRayBot import VRBot

# ---------- Augmentation Utilities ----------


LOG_FILE = f"multiagent_agumented_{INTRINSIC_WEIGHT}-agent.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        write = csv.writer(f)
        write.writerow(["episode", "agent", "score", "action"])
        
def log_espisode(episode, agent, score, a):
    with open(LOG_FILE, "a", newline="") as f:
        write = csv.writer(f)
        write.writerow([episode, agent, score, a])


def _clamp_img(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def random_crop_and_resize(img, crop_pct=AUG_CROP_PCT, out_size=(224, 224)):
    h, w = img.shape[:2]
    max_crop_h = int(h * crop_pct)
    max_crop_w = int(w * crop_pct)
    top = random.randint(0, max(0, max_crop_h))
    left = random.randint(0, max(0, max_crop_w))
    bottom = h - (max_crop_h - top)
    right = w - (max_crop_w - left)
    # guard against degenerate crop
    if bottom <= top or right <= left:
        return cv2.resize(img, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
    cropped = img[top:bottom, left:right]
    return cv2.resize(cropped, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)


def random_shift(img, shift_pct=AUG_SHIFT_PCT):
    h, w = img.shape[:2]
    max_dx = int(w * shift_pct)
    max_dy = int(h * shift_pct)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return shifted


def color_jitter(img, brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST):
    img_f = img.astype(np.float32)
    b = 1.0 + random.uniform(-brightness, brightness)
    img_f = img_f * b
    c = 1.0 + random.uniform(-contrast, contrast)
    mean = img_f.mean(axis=(0, 1), keepdims=True)
    img_f = (img_f - mean) * c + mean
    return _clamp_img(img_f)


def gaussian_noise(img, std=AUG_NOISE_STD):
    img_f = img.astype(np.float32)
    noise = np.random.normal(0, std, img_f.shape)
    img_f += noise
    return _clamp_img(img_f)


def random_cutout(img, cutout_pct=AUG_CUTOUT_PCT):
    h, w = img.shape[:2]
    cut_w = int(w * cutout_pct)
    cut_h = int(h * cutout_pct)
    if cut_w == 0 or cut_h == 0:
        return img
    x = random.randint(0, max(0, w - cut_w))
    y = random.randint(0, max(0, h - cut_h))
    img2 = img.copy()
    img2[y:y + cut_h, x:x + cut_w] = 127
    return img2


def augment_observation(obs, out_size=(224, 224), prob=AUG_PROB):
    """Return augmented copy of obs (H,W,C uint8). Apply each transform stochastically."""
    if random.random() > prob:
        # no augmentation: return resized copy to ensure consistent shape
        if (obs.shape[0], obs.shape[1]) != (out_size[0], out_size[1]):
            return cv2.resize(obs, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
        return obs.copy()
    img = obs.copy()
    img = random_crop_and_resize(img, crop_pct=AUG_CROP_PCT, out_size=out_size)
    if random.random() < 0.5:
        img = random_shift(img, shift_pct=AUG_SHIFT_PCT)
    if random.random() < AUG_FLIP_PROB:
        img = cv2.flip(img, 1)
    if random.random() < 0.8:
        img = color_jitter(img, brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST)
    if random.random() < 0.5:
        img = gaussian_noise(img, std=AUG_NOISE_STD)
    if random.random() < 0.4:
        img = random_cutout(img, cutout_pct=AUG_CUTOUT_PCT)
    return img


# ---------- Live plotter ----------

def live_plotter(queue, num_agents=2):
    plt.ion()
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))

    intrinsic_vals = [[] for _ in range(num_agents)]
    heuristic_vals = [[] for _ in range(num_agents)]
    reward_vals = [[] for _ in range(num_agents)]
    curiosity_vals = [[] for _ in range(num_agents)]
    risk_vals = [[] for _ in range(num_agents)]

    colors = ["orange", "blue"]
    labels = [f"Agent {i}" for i in range(num_agents)]

    while True:
        msg = queue.get()
        if msg is None:  # Shutdown signal
            break

        agent_index, intrinsic, heuristic, reward_final, curiosity, risk = msg
        intrinsic_vals[agent_index].append(intrinsic)
        heuristic_vals[agent_index].append(heuristic)
        reward_vals[agent_index].append(reward_final)
        curiosity_vals[agent_index].append(curiosity)
        risk_vals[agent_index].append(risk)

        for ax in axs:
            ax.cla()

        for i in range(num_agents):
            axs[0].plot(intrinsic_vals[i], color=colors[i], label=labels[i])
            axs[1].plot(heuristic_vals[i], color=colors[i], label=labels[i])
            axs[2].plot(reward_vals[i], color=colors[i], label=labels[i])
            axs[3].plot(curiosity_vals[i], color=colors[i], label=labels[i])
            axs[4].plot(risk_vals[i], color=colors[i], label=labels[i])

        axs[0].set_title("Intrinsic Reward")
        axs[1].set_title("Heuristic Reward")
        axs[2].set_title("Total Reward")
        axs[3].set_title("Curiosity")
        axs[4].set_title("Macro-Strategic Risk R_MS")

        for ax in axs:
            ax.legend(loc="upper left")

        plt.pause(0.05)

    plt.ioff()
    plt.show()


# ---------- Multi-game SC2 thread ----------
class MultiGameThread(Thread):
    def __init__(self, map_name: str, num_agents: int = 2, realtime: bool = False):
        super().__init__(daemon=True)
        self.map_name = map_name
        self.num_agents = num_agents
        self.action_ins = [Queue() for _ in range(num_agents)]
        self.result_outs = [Queue() for _ in range(num_agents)]
        self._running = False

    def run(self):
        bots = [
            Bot(Race.Protoss,
                VRBot(action_in=self.action_ins[i], result_out=self.result_outs[i], player_index=i))
            for i in range(self.num_agents)
        ]
        self._running = True
        print("Starting multi-agent SC2 match...")
        run_game(maps.get(self.map_name), bots, realtime=False)
        self._running = False

    def stop(self):
        self._running = False
        for q in self.action_ins:
            q.put("quit")


# ---------- Per-agent environment ----------
class PerAgentEnv:
    """A thin environment wrapper around the communication queues with VRBot."""

    def __init__(self, game_thread: MultiGameThread, agent_index: int):
        self.game = game_thread
        self.agent_index = agent_index
        self.action_in = self.game.action_ins[agent_index]
        self.result_out = self.game.result_outs[agent_index]
        self.observation_space = np.zeros(OBS_SHAPE, dtype=np.uint8)
        self.action_space = np.arange(N_ACTIONS)
        self.last_obs = None
        self.current_episode_reward = 0.0

    def step(self, action):
        # Send action and block until bot returns the step result
        self.action_in.put(int(action))
        out = self.result_out.get()  # blocking
        obs = out['observation'].astype(np.uint8)
        reward = float(out.get('reward', 0.0))
        done = bool(out.get('done', False))
        info = out.get('info', {}) or {}

        self.last_obs = obs
        self.current_episode_reward += reward
        return obs, reward, done, info

    def reset(self):
        # ask bot to reset -- VRBot should handle a 'reset' action string
        self.action_in.put("reset")
        out = self.result_out.get()
        obs = out['observation'].astype(np.uint8)
        self.last_obs = obs
        self.current_episode_reward = 0.0
        return obs


# ---------- Curiosity and risk ----------

def curiosity_bonus(obs, next_obs):
    diff = np.mean(np.square(next_obs.astype(np.float32) - obs.astype(np.float32)))
    return diff / (255.0 * 255.0)


def macro_strategic_risk(value_i, value_j, max_value=MAX_GAME_VALUE, lambd=LAMBDA, sigma=0.2):
    value_i, value_j = float(value_i), float(value_j)
    adv = (value_i - value_j) / max_value
    risk_ms = np.exp(- (adv ** 2) / (2.0 * (sigma ** 2)))
    return float(risk_ms)


# ---------- Simple CNN Actor-Critic ----------
class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # compute conv output dim with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, OBS_SHAPE[0], OBS_SHAPE[1])
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU()
        )
        self.policy_logits = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x: np.uint8 H W C -> convert to torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
            if x.ndim == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
            elif x.ndim == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        features = self.conv(x)
        features = self.fc(features)
        logits = self.policy_logits(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


# ---------- PPO buffer and GAE ----------
Transition = namedtuple('Transition', ['obs', 'action', 'logp', 'reward', 'done', 'value'])


class RolloutBuffer:
    def __init__(self):
        self.storage: List[Transition] = []

    def add(self, *args):
        self.storage.append(Transition(*args))

    def clear(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)


def compute_gae(transitions: List[Transition], last_value: float, gamma: float, lam: float):
    obs = np.stack([t.obs for t in transitions])
    actions = np.array([t.action for t in transitions])
    logps = np.array([t.logp for t in transitions])
    values = np.array([t.value for t in transitions] + [last_value])

    rewards = np.array([t.reward for t in transitions])
    dones = np.array([t.done for t in transitions])

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1.0 - dones[step]) * gae
        advantages[step] = gae
    returns = advantages + values[:-1]
    return {
        'obs': obs,
        'actions': actions,
        'logps': logps,
        'advantages': advantages,
        'returns': returns,
        'values': values[:-1]
    }


# ---------- Agent wrapper that contains network + optimizer ----------
class PPOAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.net = ActorCriticCNN(N_ACTIONS).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.buffer = RolloutBuffer()

    def get_action(self, obs: np.ndarray):
        # obs H W C (uint8)
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
        logits, value = self.net(obs_t)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        logp = m.log_prob(torch.tensor(action).to(DEVICE)).item()
        return action, logp, value.item()

    def update(self, batch: Dict):
        obs = torch.from_numpy(batch['obs']).float().to(DEVICE)
        # obs is batch H W C -> convert to batch C H W
        if obs.ndim == 4 and obs.shape[-1] == 3:
            obs = obs.permute(0, 3, 1, 2)
        actions = torch.from_numpy(batch['actions']).long().to(DEVICE)
        old_logps = torch.from_numpy(batch['logps']).float().to(DEVICE)
        advantages = torch.from_numpy(batch['advantages']).float().to(DEVICE)
        returns = torch.from_numpy(batch['returns']).float().to(DEVICE)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = obs.shape[0]
        idxs = np.arange(dataset_size)

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, MINIBATCH_SIZE):
                mb_idx = idxs[start:start + MINIBATCH_SIZE]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                logits, values = self.net(mb_obs)
                dist = F.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(dist)
                mb_logps = m.log_prob(mb_actions)
                entropy = m.entropy().mean()

                ratio = torch.exp(mb_logps - mb_old_logps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_ret)

                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))


# ---------- Training coordinator (with augmentation) ----------

def train_two_agents():
    # Start SC2 thread
    game_thread = MultiGameThread(MAP_NAME, num_agents=NUM_AGENTS, realtime=False)
    try:
        game_thread.start()
    except Exception:
        print("Could not start SC2 thread; ensure SC2 is installed and configured.")
        return

    # Wait until bots are ready to accept messages
    time.sleep(3.0)

    envs = [PerAgentEnv(game_thread, i) for i in range(NUM_AGENTS)]
    agents = [PPOAgent(i) for i in range(NUM_AGENTS)]

    # initial reset: asks bots for the initial observation
    observations = [env.reset() for env in envs]

    episode_rewards = [0.0 for _ in range(NUM_AGENTS)]
    episode_lengths = [0 for _ in range(NUM_AGENTS)]

    global_step = 0

    plot_queue = mp.Queue()
    plot_process = mp.Process(target=live_plotter, args=(plot_queue, NUM_AGENTS))
    plot_process.start()


    for update in range(1, TOTAL_UPDATES + 1):
        # collect rollouts for STEPS_PER_UPDATE steps in lockstep
        for step in range(STEPS_PER_UPDATE):
            actions = [None] * NUM_AGENTS
            logps = [None] * NUM_AGENTS
            values = [None] * NUM_AGENTS

            # 1) augment observations and get actions for all agents
            augmented_obs = [augment_observation(observations[i]) for i in range(NUM_AGENTS)]
            for i in range(NUM_AGENTS):
                a, logp, val = agents[i].get_action(augmented_obs[i])
                actions[i] = a
                logps[i] = logp
                values[i] = val

            # 2) step both envs in lockstep (send both actions to the SC2 bots)
            next_obs = [None] * NUM_AGENTS
            rewards = [0.0] * NUM_AGENTS
            dones = [False] * NUM_AGENTS
            infos = [None] * NUM_AGENTS

            for i in range(NUM_AGENTS):
                obs_i, r_i, done_i, info_i = envs[i].step(actions[i])
                next_obs[i] = obs_i
                rewards[i] = float(r_i)
                dones[i] = bool(done_i)
                infos[i] = info_i or {}

            # 3) compute intrinsic bonuses and final reward per agent and store augmented obs
            for i in range(NUM_AGENTS):
                j = 1 - i
                self_val = infos[i].get('self_value', MAX_GAME_VALUE / 2)
                opp_val = infos[i].get('opponent_value', MAX_GAME_VALUE / 2)

                R_MS = macro_strategic_risk(self_val, opp_val)
                T_s = 1.0 - R_MS
                # curiosity uses raw observations (not augmented) to reflect environment delta
                R_cur = curiosity_bonus(observations[i], next_obs[i])
                intrinsic_bonus = (T_s * ALPHA * R_cur) - (BETA * R_MS)
                heuristic_reward = infos[i].get('heuristic_reward', float(rewards[i]))
                reward_final = INTRINSIC_WEIGHT * intrinsic_bonus + (1 - INTRINSIC_WEIGHT) * heuristic_reward

                # store transition using augmented observation (agents train on augmented data)
                agents[i].buffer.add(augmented_obs[i], actions[i], logps[i], reward_final, dones[i], values[i])

                episode_rewards[i] += reward_final
                episode_lengths[i] += 1
                log_espisode(update, i,episode_rewards, a)

                # send values for plotting
                try:
                    plot_queue.put((i, intrinsic_bonus, heuristic_reward, reward_final, R_cur, R_MS))
                except Exception:
                    pass

            observations = next_obs
            global_step += 1

            # handle episode termination: if any agent signals done, reset both to keep sync
            if any(dones):
                # logging
                for i in range(NUM_AGENTS):
                    if dones[i]:
                        print(f"[update {update}] Agent {i} episode done | length={episode_lengths[i]} | reward={episode_rewards[i]:.3f}")
                        # write to csv per agent
                        fname = os.path.join(MODELS_DIR, f"agent_{i}_rewards.csv")
                        with open(fname, 'a', newline='') as f:
                            w = csv.writer(f)
                            if os.stat(fname).st_size == 0:
                                w.writerow(["TotalEpisodeReward"])
                            w.writerow([episode_rewards[i]])
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0

                # reset both envs to keep them synchronized
                observations = [env.reset() for env in envs]

        # end rollout collection for this update

        # For each agent, compute GAE and perform PPO updates
        for i in range(NUM_AGENTS):
            # estimate last value for bootstrapping
            last_obs = observations[i]
            with torch.no_grad():
                t = torch.from_numpy(last_obs).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
                _, last_value = agents[i].net(t)
                last_value = last_value.item()

            transitions = agents[i].buffer.storage
            if len(transitions) == 0:
                continue

            batch = compute_gae(transitions, last_value, GAMMA, LAM)
            # update agent
            agents[i].update(batch)
            agents[i].buffer.clear()

        # logging & save
        if update % LOG_INTERVAL == 0:
            print(f"Update {update}/{TOTAL_UPDATES} | global_steps={global_step}")

        if update % SAVE_INTERVAL == 0:
            for i in range(NUM_AGENTS):
                path = os.path.join(MODELS_DIR, f"agent_{i}_update_{update}.pth")
                agents[i].save(path)
            print(f"Saved models at update {update}")

    # cleanup
    try:
        plot_queue.put(None)
    except Exception:
        pass
    plot_process.join()

    print("Training finished. Stopping game thread...")
    game_thread.stop()


if __name__ == '__main__':
    train_two_agents()
