import numpy as np
import torch
import gym
from torch import nn
import time
import wandb
from tqdm import tqdm


env = gym.make("CartPole-v1")
env = gym.wrappers.RecordVideo(env, 'video_A2C_CartePole', episode_trigger = lambda x: x % 2 == 0)
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
GAMMA = 0.99
EPISODES = 5002
ARRAY_OF_SEED = [1,2,3,4,5] 

def t(x): return torch.from_numpy(x).float()
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax()
        )
    
    def forward(self, X):
        return self.model(X)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, X):
        return self.model(X)


class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

def train(memory, q_val):
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))
    
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + GAMMA *q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val
        
    advantage = torch.Tensor(q_vals) - values
    
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()
    
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()


      
for seed in ARRAY_OF_SEED:
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    run = wandb.init(project="CartPole-v1-AC",reinit=True)
    wandb.run.name = "AC agent : " + wandb.run.id
    wandb.run.save()
    torch.device("cuda")

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    memory = Memory()
  
    wandb.config = {
        "gamma": GAMMA,
        "learning rate": 1e-3,
        "hidden layers": 64,
        "EPISODES": EPISODES,
        "optimizer": "Adam",
        "device" : "cuda",
        "seed": [str(x) for x in ARRAY_OF_SEED]
    }

    total_start_time = time.time()
    for i in tqdm(range(EPISODES)):
        done = False
        total_reward = 0
        state = env.reset()
        steps = 0
        while not done:
            probs = actor(t(state))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.detach().data.numpy())
            
            total_reward += reward
            steps += 1
            memory.add(dist.log_prob(action), critic(t(state)), reward, done)
            
            state = next_state
            
        last_q_val = critic(t(next_state)).detach().data.numpy()
        train(memory, last_q_val)
        memory.clear()
        wandb.log({"Reward": total_reward})
    total_stop_time = time.time()
    print("the episode took : ", (total_stop_time - total_start_time))
    run.finish()

