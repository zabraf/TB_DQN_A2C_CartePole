import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import time
import copy

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordVideo(env, 'video_DQN_CartePole', episode_trigger = lambda x: x %  500 == 0)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

EPISODES = 5002
LEARNING_RATE = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
FC1_DIMS = 256
DEVICE = torch.device("cuda")
ARRAY_OF_SEED = [1,2,3,4,5]

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = action_space

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ExperienceReplay:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self):
        self.memory = ExperienceReplay()
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network()
        self.target_net = copy.deepcopy(self.network)
        self.network_sync_counter = 0
        self.network_sync_freq = 5

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return env.action_space.sample()
        
        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        if(self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.network.state_dict())
            self.network_sync_counter = 0

        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)
        

        q_values = self.network(states)
        next_q_values = self.target_net(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + GAMMA * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        self.network_sync_counter += 1   

for seed in ARRAY_OF_SEED:
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    run = wandb.init(project="CartPole-v1-DQN",reinit=True)
    wandb.run.name = "DQN agent : " + wandb.run.id

    wandb.config = {
        "episodes": EPISODES,
        "learning_rate": LEARNING_RATE,
        "mem_size": MEM_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "exploration_max": EXPLORATION_MAX,
        "exploration_decay": EXPLORATION_DECAY,
        "exploration_min": EXPLORATION_MIN,
        "fc1_dims": FC1_DIMS,
        "device": "cuda",
        "network sync frequency": 5,
        "seed": [str(x) for x in ARRAY_OF_SEED]
    }
    agent = DQN_Solver()


    total_start_time = time.time()
    for i in tqdm(range(1, EPISODES + 1)):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        score = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            state_ = np.reshape(state_, [1, observation_space])
            agent.memory.add(state, action, reward, state_, done)
            agent.learn()
            state = state_
            score += reward

        wandb.log({"Reward": score})
    total_stop_time = time.time()
    print("the episode took : ", (total_stop_time - total_start_time))
    run.finish()