import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import rl_utils
from ppo import *

# 1. Behaviour Clone

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)

def sample_expert_data(n_episode):
    states = []
    actions = []
    for episode in range(n_episode):
        state = env.reset()
        done = False
        while not done:
            action = ppo_agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state,reward,done,_ = env.step(action)
            state = next_state
    return np.array(states) , np.array(actions)



class BehaviorClone:
    def __init__(self,state_dim,hidden_dim,action_dim,lr):
        self.policy = PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr = lr)
    
    def learn(self,states,actions):
        states = torch.tensor(states,dtype = torch.float).to(device)
        actions = torch.tensor(actions).view(-1,1).to(device)
        log_probs = torch.log(self.policy(states).gather(1,actions))
        bc_loss = torch.mean(-log_probs)
        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()
    def take_action(self,state):
        state = torch.tensor([state],dtype = torch.float).to(device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    

def test_agent(agent,env,n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state,reward, done,_ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)




# 2. GAIL Algorithm

class Discriminator(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Discriminator,self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)
    def forward(self,x,a):
        cat = torch.cat([x,a],dim = 1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))

class GAIL:
    def __init__(self,agent,state_dim,action_dim,hidden_dim,lr_d):
        self.discriminator = Discriminator(state_dim,hidden_dim,action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr = lr_d)
        self.agent = agent

    def learn(self,expert_s,expert_a,agent_s,agent_a,next_s,next_a,dones):
        expert_states = torch.tensor(expert_s,dtype = torch.float).to(device)
        expert_actions = torch.tensor(expert_a).to(device)
        agent_states = torch.tensor(agent_s,dtype = torch.float).to(device)
        agent_actions = torch.tensor(agent_a).to(device)
        expert_actions = F.one_hot(expert_actions,num_classes = 2).float()
        agent_actions = F.one_hot(agent_actions,num_classes=2).float()
        expert_prob = self.discriminator(expert_states,expert_actions)
        agent_prob = self.discriminator(agent_states,agent_actions)
        discriminator_loss = nn.BCELoss()(agent_prob,torch.ones_like(agent_prob)) + nn.BCELoss()(expert_prob,torch.zeros_like(expert_prob))
        self.discriminator.zero_grad()
        discriminator_loss.backward()
        self.discriminator.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards':rewards,
            'next_states':next_s,
            'dones':dones
        }
        self.agent.update(transition_dict)

        
