import gymnasium as gym
from functorch.dim import dims
from mypy.plugins.dataclasses import dataclass_class_maker_callback

from lib_pong import dqn_model
from lib_pong import wrappers
from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

DEFAULT_ENV_NAME= "PongNoFrameskip-v4"
MEAN_REWARD_BOUND=19
GAMMA=0.99
BATCH_SIZE=12
REPLAY_SIZE=10000  #max capacity of buffer
LEARNING_RATE=1e-4
SYNC_TARGET_FRAMES=1000
REPLAY_START_SIZE=10000 #The count of frames we wait for before starting training to populate the replay buffer
EPSILON_DECAY_LAST_FRAME=150000
EPSILON_START=1.0
EPSILON_FINAL=0.01

State= np.ndarray
Action= int
BatchTensors= tt.Tuple[
    torch.ByteTensor, #current state
    torch.LongTensor, #actions
    torch.Tensor, #rewards
    torch.BoolTensor, #done or trunc
    torch.ByteTensor, #next state
]
@dataclass
class Experience:
    state:State
    action:Action
    reward:float|int
    done_trunc: bool
    new_state: State

class ExperienceBuffer:
    def __init__(self,capacity:int ):
        self.buffer = collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self,experience:Experience):
        self.buffer.append(experience)
    def sample(self,batch_size:int )->tt.List[Experience]:
        indices= np.random.choice(len(self),batch_size,replace=False)
        return[self.buffer[idx] for idx in indices]

class Agent:
    def __init__(self,env: gym.Env,exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray]=None
        self._reset()
    def _reset(self):
        self.state,_=self.env.reset()
        self.total_reward=0.0

    @torch.no_grad()
    def play_step(self,net:dqn_model.DQN,device: torch.device, epsilon: float|int=0.0)-> tt.Optional[float|int]:
        done_reward= None
        if np.random.random() < epsilon:
            action=self.env.action_space.sample()
        else:
            state_v= torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v=net(state_v)
            _,act_v=torch.max(q_vals_v,dim=1)
            action=int(act_v.item())
        new_state,reward,is_done,is_trunc,_= self.env.step(action)
        self.total_reward+=reward

        exp=Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            done_trunc=is_trunc or is_done,
            new_state=new_state
        )
        self.exp_buffer.append(exp)
        self.state=new_state
        if is_done or is_trunc:
            done_reward=self.total_reward
            self._reset()
        return done_reward

    def batch_to_tensors(batch: tt.List[Experience],device: torch.device)->BatchTensors:
        states,actions,rewards,dones,new_states=[],[],[],[],[]
        for e in batch:
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            dones.append(e.done_trunc)
            new_states.append(e.new_state)
        states_t=torch.as_tensor(np.asarray(states))
        actions_t= torch.LongTensor(actions)
        rewards_t= torch.FloatTensor(rewards)
        dones_t= torch.BoolTensor(dones)
        new_states_t=torch.as_tensor(np.asarray(new_states))
        return states_t.to(device), actions_t.to(device), rewards_t.to(device), dones_t.to(device), new_states_t.to(device)
    