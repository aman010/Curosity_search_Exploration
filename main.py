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

# StableBaselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# SC2 API imports
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps

# Bot
from VoidRayBot import VRBot

# Global variables to pick the right experiment and WandB project.
mapName = "AbyssalReefLE"
episode_reward_list = []

# Change the comments in the following two lines to create a new model 
model_name = f"{int(time.time())}"
#model_name = 1713701470

models_dir = f"models/{model_name}/"

# This is the thread that holds the queues and runs the game
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

# This is the environment itself where Step and Reset are defined
class QueueEnv(gym.Env):
    def __init__(self, config=None, render_mode=None): # None, "human", "rgb_array"
        super(QueueEnv, self).__init__()
        self.action_space = Discrete(constants.NUMBER_OF_ACTIONS)
        self.observation_space = constants.OBSERVATION_SPACE_ARRAY
        self.current_episode_reward = 0 
        self.rewards_file = os.path.join(models_dir, "episode_rewards.csv")  # Define el archivo CSV

    def step(self, action):
        # Send an action to the Bot
        self.gameThread.action_in.put(action)

        # Get the result
        out = self.gameThread.result_out.get()               
        observation = out["observation"].astype(np.int32)
        reward = out["reward"]
        done = out["done"]
        truncated = out["truncated"]
        info = out["info"]

        self.current_episode_reward += reward 

        if done:
            with open(self.rewards_file, 'a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["Total Episode Reward"])
                writer.writerow([self.current_episode_reward])
            self.current_episode_reward = 0 
        observation = np.clip(observation, 0, np.inf).astype(np.int32)
        return observation, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("--- RESETTING ENVIRONMENT ---")
        time.sleep(5)
        observation = np.zeros((224, 224, 3), dtype=np.uint8)
        info = {}
        self.gameThread = GameThread()
        self.gameThread.start()
        return observation, info
        
# Inside your main script (or environment file)

# ... (Existing imports)
'''import constants 
from curiosity_module import IntrinsicCuriosityModule # New Import!'''
# ...

# This is the environment itself where Step and Reset are defined
'''class QueueEnv(gym.Env):
    def __init__(self, config=None, render_mode=None, icm=None): # Added icm argument
        super(QueueEnv, self).__init__()
        self.action_space = Discrete(constants.NUMBER_OF_ACTIONS)
        self.observation_space = constants.OBSERVATION_SPACE_ARRAY
        self.current_episode_reward = 0 
        self.rewards_file = os.path.join(models_dir, "episode_rewards.csv")

        # ICM Setup
        self.icm = icm # The ICM module
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
        # 2. Calculate Intrinsic Reward (r_i)
        # -----------------------------------------------
        intrinsic_reward = 0.0
        if self.icm is not None and state_t is not None:
            # ICM needs the action taken, state_t, and next_state_t
            intrinsic_reward = self.icm.get_intrinsic_reward(
                state=state_t, 
                next_state=observation, 
                action=action
            )
        
        # Calculate the total reward
        total_reward = extrinsic_reward + constants.BETA * intrinsic_reward

        # -----------------------------------------------
        
        self.current_episode_reward += total_reward # Use total_reward for logging
        self.last_observation = observation.copy() # Store s_{t+1} for the next step's s_t

        if done:
            # ... (Existing reward logging logic, use self.current_episode_reward)
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
        observation = np.zeros((224, 224, 3), dtype=np.uint8)
        info = {}
        self.gameThread = GameThread()
        self.gameThread.start()
        
        # Initialize last_observation on reset
        self.last_observation = observation.copy() 
        
        return observation, info'''

# ... (make_env remains the same)


def make_env():
    def _init():
        env = QueueEnv()
        return env
    return _init

def train_ppo():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    num_envs = constants.NUMBER_OF_CONCURRENT_EXECUTIONS
    #env = SubprocVecEnv([make_env() for i in range(num_envs)])
    env = DummyVecEnv([make_env() for i in range(1)])
    model_path = os.path.join(models_dir, "model.zip")

    if os.path.exists(model_path):
        print("Loading existing model")
        model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=f"./ppo_tb_{model_name}")
    else:
        print("Creating new model")
        model = PPO('CnnPolicy', env, n_steps=256, batch_size=32, verbose=2, tensorboard_log=f"./ppo_tb_{model_name}")

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir,
                                             name_prefix='ppo_model')

    iters = 0
    while iters < constants.NUMBER_OF_ITERATIONS:
        print(f"On iteration: {iters}")
        iters += 1
        model.learn(total_timesteps=constants.TIMESTEPS, reset_num_timesteps=False,
                    tb_log_name=f"PPO_run_tb_{model_name}", callback=checkpoint_callback, progress_bar=True)
        model.save(os.path.join(models_dir, "model"))

    env.close()
    
# Inside your main script (or training file)

# Inside your main script (or training file)

'''def train_ppo():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # -----------------------------------------------
    # 3. Instantiate and share the ICM
    # -----------------------------------------------
    # ICM is instantiated once and passed to all environments for reward calculation
    # NOTE: To fully train the ICM, you MUST either:
    # 1. Subclass the SB3 PPO policy and add the ICM loss to the total objective. (Hard)
    # 2. Train the ICM separately using the collected rollouts in a custom callback. (Medium)
    #
    # For simplicity, we are only implementing the reward calculation part here.
    icm_module = IntrinsicCuriosityModule(
        observation_space=constants.OBSERVATION_SPACE_ARRAY,
        action_space=Discrete(constants.NUMBER_OF_ACTIONS),
        feature_dim=constants.ICM_ENCODER_DIM
    )
    
    def make_env(icm_instance):
        def _init():
            # Pass the ICM instance to the environment
            env = QueueEnv(icm=icm_instance) 
            return env
        return _init

    num_envs = constants.NUMBER_OF_CONCURRENT_EXECUTIONS
    #env = SubprocVecEnv([make_env() for i in range(num_envs)])
    
    # Use the shared ICM instance when creating the environments
    env = DummyVecEnv([make_env(icm_module) for i in range(1)]) 

    # ... (Model loading/creation remains similar, but now PPO receives ICM rewards)

    if os.path.exists(model_path):
    	pass
        # ... loading logic ...
    else:
        print("Creating new model")
        # PPO's CnnPolicy will handle the visual inputs
        model = PPO('CnnPolicy', env, n_steps=256, batch_size=32, verbose=2, tensorboard_log=f"./ppo_tb_{model_name}")

    # ... (Training loop remains the same)'''
    
    # Note: If you don't fully train the ICM, the 'curiosity' will decrease slowly
    # as the PPO agent's value function implicitly trains the forward model. 
    # For a *proper* implementation, you need to add the ICM's loss to the PPO optimization step.
if __name__ == "__main__":
    train_ppo()
    
