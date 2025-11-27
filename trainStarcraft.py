# train_starcraft.py

# General imports
import numpy as np
import time
import csv
import os
import constants
from queue import Queue
from threading import Thread

# OpenAI Gymnasium imports
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, Box
import torch
import torch.nn.functional as F

# StableBaselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples

# SC2 API imports
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps

# Local imports
from VoidRayBot import VRBot # Assuming this exists
from curiosity_module import IntrinsicCuriosityModule # New Import!

# Global variables
mapName = "AbyssalReefLE"
model_name = f"{int(time.time())}"
models_dir = f"models/{model_name}/"
model_path = os.path.join(models_dir, "model.zip")

# Ensure the ICM model is available to the callback
global icm_module_global 
icm_module_global = None 

# --------------------------------------------------------------------------
# 1. ICM Training Callback 
# --------------------------------------------------------------------------

class ICMCallback(BaseCallback):
    """
    A custom callback to train the Intrinsic Curiosity Module (ICM) 
    using the collected rollouts from the PPO agent.
    """
    def __init__(self, icm_module, verbose=0):
        super(ICMCallback, self).__init__(verbose)
        self.icm = icm_module
        self.device = "cpu"

    def _on_rollout_end(self) -> None:
        """Called after rollout ends, before the policy is updated."""
        
        # ... (buffer type check remains the same)

        rollout_data = self.model.rollout_buffer

        # Move rollout data to device
        # NOTE: obs has shape (n_steps, n_envs, H, W, C)
        obs = torch.as_tensor(rollout_data.observations, device=self.device)
        actions = torch.as_tensor(rollout_data.actions, device=self.device)
        
        n_steps = self.model.n_steps
        n_envs = self.model.n_envs
        buffer_size = n_steps * n_envs
        H, W, C = constants.OBSERVATION_SPACE_ARRAY.shape # (224, 224, 3)

        # ----------------------------------------------------------------------
        # CRITICAL FIX 1: Reshape 5-D observation tensor to 4-D (N_total, H, W, C)
        # ----------------------------------------------------------------------
        obs = torch.as_tensor(obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        # Compute the real number of transitions available
        num_samples = obs.numel() // (H * W * C)
        obs = obs.reshape(num_samples, H, W, C)
        actions = actions.reshape(-1)
        if actions.shape[0] > num_samples:
        	actions = actions[:num_samples]
        next_obs_np = np.roll(obs.cpu().numpy(), -1, axis=0)
        next_obs = torch.as_tensor(next_obs_np, device=self.device)

        # ----------------------------------------------------------------------
        # CRITICAL FIX 2: Crop to correct buffer size (removing the extra n_envs steps)
        # ----------------------------------------------------------------------
        #obs = obs[:buffer_size]
        #next_obs = next_obs[:buffer_size]
        #actions = actions[:buffer_size]
        
        # The actions should already be 1D due to the reshape and squeeze
        # if actions.ndim > 1:
        #     actions = actions.squeeze(-1) # This is now redundant

        # Forward pass through the ICM (now receiving guaranteed 4D tensors (N, H, W, C))
        pred_action_logits, pred_next_state, phi_next_state_actual, intrinsic_reward = self.icm(
            obs, next_obs, actions
        )

        

        # ------------------------------------------
        # Calculate ICM Losses
        # ------------------------------------------
        
        # 1. Inverse Dynamics Loss (L_I): Train the encoder (phi) and inverse model
        # The cross-entropy loss expects action to be long tensor of class indices
        L_I = F.cross_entropy(pred_action_logits, actions.long())

        # 2. Forward Dynamics Loss (L_F): Train the forward model
        L_F = F.mse_loss(pred_next_state, phi_next_state_actual.detach()) # Detach phi_next_state_actual

        # Total ICM Loss
        # L_ICM = (1 - BETA) * L_I + BETA * L_F (using constants.ICM_LOSS_WEIGHT for inverse loss)
        L_ICM = (1.0 - constants.ICM_LOSS_WEIGHT) * L_I + constants.ICM_LOSS_WEIGHT * L_F
        
        # ------------------------------------------
        # Update ICM Networks
        # ------------------------------------------
        self.icm.optimizer.zero_grad()
        L_ICM.backward()
        self.icm.optimizer.step()
        
        if self.verbose > 0:
            print(f"ICM Update: L_I={L_I.item():.4f}, L_F={L_F.item():.4f}, L_ICM={L_ICM.item():.4f}")

    def _on_step(self) -> bool:
        return True

# --------------------------------------------------------------------------
# 2. Environment Definitions (QueueThread and QueueEnv)
# --------------------------------------------------------------------------

class GameThread(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.action_in = Queue()
        self.result_out = Queue()
 
    def run(self) -> None:
        self.bot = VRBot(action_in=self.action_in, result_out=self.result_out)
        print("starting game.")
        result = run_game(  # run_game is a function that runs the game.
            maps.get(mapName), # the map we are playing on
            [Bot(Race.Protoss, self.bot), # runs our coded bot, and we pass our bot object 
            Computer(Race.Terran, Difficulty.Medium)], # runs a pre-made computer agent, with a hard difficulty.
            realtime=True, # When set to True, the agent is limited in how long each step can take to process.
        )

class QueueEnv(gym.Env):
    def __init__(self, config=None, render_mode=None, icm=None):
        super(QueueEnv, self).__init__()
        self.action_space = Discrete(constants.NUMBER_OF_ACTIONS)
        self.observation_space = constants.OBSERVATION_SPACE_ARRAY
        self.current_episode_reward = 0 
        self.rewards_file = os.path.join(models_dir, "episode_rewards.csv")

        # ICM Setup
        self.icm = icm 
        self.last_observation = None # To store s_t for ICM calculation

    def step(self, action):
    	# Store the current state (s_t) before taking the action
    	state_t = self.last_observation.copy() if self.last_observation is not None else None

    	# Send an action to the Bot
    	self.gameThread.action_in.put(action)

    	# Get the result (s_{t+1}, r_e, done, etc.)
    	out = self.gameThread.result_out.get()      
    	observation = out["observation"].astype(np.int32)
    	extrinsic_reward = out["reward"] # r_e
    	done = out["done"]
    	truncated = out["truncated"]
    	info = out["info"]

    	# -----------------------------------------------
    	# Calculate Intrinsic Reward (r_i)
    	# -----------------------------------------------
    	intrinsic_reward = 0.0
    	if self.icm is not None and state_t is not None:
        
        	# --- CRITICAL FIX: Add Batch Dimension and Convert to Tensor ---
       	 # The environment returns (H, W, C). ICM requires (B, H, W, C) where B=1.
        
        	# NOTE: Assumes self.device is defined in the environment class
        	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        # state_t is the observation at time t (s_t), currently (H, W, C)
        	state_tensor = torch.as_tensor(state_t, device=device).unsqueeze(0)
        
        # observation is the observation at time t+1 (s_{t+1}), currently (H, W, C)
        	next_state_tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
        
        # action is a single integer, convert to tensor and unsqueeze (B=1)
        	action_tensor = torch.as_tensor([action], device=device)
        
        # ICM needs the action taken, state_t, and next_state_t
        	intrinsic_reward = self.icm.get_intrinsic_reward(
            	state=state_tensor, 
            	next_state=next_state_tensor, 
            	action=action_tensor
        		)
 
    # Calculate the total reward
    	total_reward = extrinsic_reward + constants.BETA * intrinsic_reward

    # -----------------------------------------------
     
    	self.current_episode_reward += total_reward
    	self.last_observation = observation.copy() # Store s_{t+1} for the next step's s_t

    	if done:
        	# ... (File writing and reset logic remains the same)
        	with open(self.rewards_file, 'a', newline='') as file:
             		writer = csv.writer(file)
             		if file.tell() == 0:
                 		writer.writerow(["Total Episode Reward"])
             		writer.writerow([self.current_episode_reward])
        	self.current_episode_reward = 0 
        	self.last_observation = None # Reset last observation
         
    	observation = np.clip(observation, 0, np.inf).astype(np.int32)
    # Return total_reward
    	return observation, total_reward, done, truncated, info
    	
    def reset(self, *, seed=None, options=None):
        print("--- RESETTING ENVIRONMENT ---")
        time.sleep(5)
        # The reset observation must match the expected observation_space shape and type
        observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype) 
        info = {}
        self.gameThread = GameThread()
        self.gameThread.start()
        
        # Initialize last_observation on reset
        self.last_observation = observation.copy() 
        
        return observation, info


def make_env(icm_instance):
    """Factory function to create a QueueEnv with the shared ICM instance."""
    def _init():
        env = QueueEnv(icm=icm_instance) 
        return env
    return _init

# --------------------------------------------------------------------------
# 3. Main Training Function
# --------------------------------------------------------------------------

def train_ppo():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # -----------------------------------------------
    # Instantiate and share the ICM
    # -----------------------------------------------
    # The ICM module will be trained on the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icm_module = IntrinsicCuriosityModule(
        observation_space=constants.OBSERVATION_SPACE_ARRAY,
        action_space=Discrete(constants.NUMBER_OF_ACTIONS)
    ).to(device)
    
    # Create the environment vector
    num_envs = constants.NUMBER_OF_CONCURRENT_EXECUTIONS
    # Use the shared ICM instance when creating the environments
    # Using DummyVecEnv for simplicity with one process
    env = DummyVecEnv([make_env(icm_module) for i in range(1)]) 

    # -----------------------------------------------
    # PPO Model Setup
    # -----------------------------------------------

    if os.path.exists(model_path):
        print("Loading existing model")
        model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=f"./ppo_tb_{model_name}", device=device)
    else:
        print("Creating new model")
        # PPO's CnnPolicy is used to handle the visual inputs
        model = PPO('CnnPolicy', env, n_steps=256, batch_size=32, verbose=2, 
                    tensorboard_log=f"./ppo_tb_{model_name}", device=device)
    
    # -----------------------------------------------
    # Callbacks
    # -----------------------------------------------
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir,
                                             name_prefix='ppo_model')
    
    icm_callback = ICMCallback(icm_module=icm_module)
    
    # Combine callbacks (CheckpointCallback must be first to save the model/ICM state)
    callbacks = [checkpoint_callback, icm_callback]

    # -----------------------------------------------
    # Training Loop
    # -----------------------------------------------

    iters = 0
    while iters < constants.NUMBER_OF_ITERATIONS:
        print(f"On iteration: {iters}")
        iters += 1
        model.learn(total_timesteps=constants.TIMESTEPS, reset_num_timesteps=False,
                    tb_log_name=f"PPO_run_tb_{model_name}", callback=callbacks, progress_bar=True)
        model.save(os.path.join(models_dir, "model"))

    env.close()

if __name__ == "__main__":
    train_ppo()
