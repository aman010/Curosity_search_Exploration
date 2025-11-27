import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.spaces import Discrete, Box
import constants

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, observation_space, action_space):
        super(IntrinsicCuriosityModule, self).__init__()
        
        feature_dim = constants.ICM_ENCODER_DIM
        action_embedding_dim = 10
        
        # NOTE: Using the H, W, C naming convention of Gym Box space
        H_in, W_in, C_in = observation_space.shape
        
        # 1. Feature Encoder: Shared Convolutional Layers (No Flatten/Linear yet)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(C_in, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # CORRECTION: Dynamic Size Calculation
        # -----------------------------------------------------------
        
        # 1. Create the test input tensor with the correct (B, H, W, C) shape.
        test_input_hwc = torch.zeros(1, H_in, W_in, C_in)
        
        # 2. Permute it to PyTorch's required (B, C, H, W) format.
        test_input_chw = test_input_hwc.permute(0, 3, 1, 2)
        
        # 3. Pass through CONVOLUTIONAL layers to determine output shape.
        with torch.no_grad():
            conv_out = self.conv_layers(test_input_chw)
        
        # 4. Calculate the size of the flattened tensor.
        conv_output_size = conv_out.flatten(start_dim=1).shape[1]
        
        # 5. Define the full feature encoder (Conv + Flatten + Linear Projection)
        self.encoder = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_output_size, feature_dim)
        )
        self.feature_dim = feature_dim

        # 2. Inverse Dynamics Model (Predicts action: phi(s_t), phi(s_t+1) -> a_t)
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )

        # 3. Forward Dynamics Model (Predicts next state: phi(s_t), a_t -> phi(s_t+1))
        self.action_embed = nn.Embedding(action_space.n, action_embedding_dim)
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Optimizer for ICM
        self.optimizer = torch.optim.Adam(self.parameters(), lr=constants.ICM_LEARNING_RATE)

    def _encode_state(self, state):
        # The ICM is receiving a 5D tensor, likely (N_envs, N_steps, H, W, C)
        # We must flatten the first two dimensions (Batch) to get a 4D tensor (N, H, W, C)
        
        # 1. Check for 5D and flatten the batch dimensions (N_envs * N_steps)
        if state.dim() == 5:
            # Flatten to (N_total, H, W, C)
            H, W, C = state.shape[-3:]
            state = state.reshape(-1, H, W, C)
        
        # Now 'state' is guaranteed to be 4D: (Batch, H, W, C)
        
        # 2. Permute from (Batch, H, W, C) to (Batch, C, H, W) for PyTorch CNN
        # This fixes the channel mismatch error that would happen next.
        state = state.permute(0, 3, 1, 2)
        
        phi_state = self.encoder(state.float())
        return phi_state

    def forward(self, state, next_state, action):
        # State encoding
        phi_state = self._encode_state(state)
        phi_next_state = self._encode_state(next_state)

        # 1. Inverse Dynamics (Loss L_I)
        concat_state = torch.cat((phi_state, phi_next_state), dim=1)
        pred_action_logits = self.inverse_net(concat_state)
        
        # 2. Forward Dynamics (Intrinsic Reward r_i and Loss L_F)
        embedded_action = self.action_embed(action.long().view(-1))
        concat_input = torch.cat((phi_state, embedded_action), dim=1)
        pred_next_state = self.forward_net(concat_input)

        # Intrinsic Reward (r_i): Mean Squared Error of the Forward Model
        # NOTE: phi_next_state must be detached here so the error does not flow back
        # and update the feature encoder via the forward loss (L_F).
        intrinsic_reward = 0.5 * torch.sum(
            (pred_next_state - phi_next_state.detach())**2, dim=1
        )
        
        return pred_action_logits, pred_next_state, phi_next_state, intrinsic_reward

    def get_intrinsic_reward(self, state, next_state, action):
        # Use for a single step calculation in the environment
        with torch.no_grad():
            state_t = torch.as_tensor(state, device='cpu').unsqueeze(0)
            next_state_t = torch.as_tensor(next_state, device='cpu').unsqueeze(0)
            action_t = torch.as_tensor(action, device='cpu').unsqueeze(0)
            
            _, _, _, intrinsic_reward = self.forward(
                state_t, 
                next_state_t, 
                action_t
            )
        # Return the scalar reward
        return intrinsic_reward.cpu().numpy()[0]
