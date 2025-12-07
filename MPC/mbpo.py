import gym
from collections import namedtuple
import itertools
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import collections
import random
import matplotlib.pyplot as plt

device = torch.device("cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound):
        super(PolicyNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim,action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim,action_dim)
        self.action_bound = action_bound
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu,std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action,log_prob


class QValueNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(QValueNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,1)
    
    def forward(self,x,a):
        cat = torch.cat([x,a],dim = -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class SAC:
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device):
        self.device = device
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim,action_bound).to(device)
        self.critic_1 = QValueNet(state_dim,hidden_dim,action_dim).to(device)
        self.critic_2 = QValueNet(state_dim,hidden_dim,action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim,hidden_dim,action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim,hidden_dim,action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),lr = critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_1.parameters(),lr = critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01),dtype = torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr = alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
    
    def take_action(self,state):
        state = torch.tensor([state],dtype = torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.tiem()]
    def calc_target(self,rewards,next_states,dones):
        next_actions,log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states,next_actions)
        q2_value = self.target_critic_2(next_states,next_actions)
        next_value = torch.min(q1_value,q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value*(1-dones)
        return td_target

    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype = torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],dtype = torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype = torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'],dtype = torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype = torch.float).view(-1,1).to(self.device)
        rewards = (rewards + 8.0) / 8,0 #Reshape the rewards
    
        td_target = self.calc_target(rewards,next_states,dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states,actions)))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states,actions)))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions,log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states,new_actions)
        q2_value = self.critic_2(states,new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value,q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        self.soft_update(self.critic_1,self.target_critic_1)
        self.soft_update(self.critic_2,self.target_critic_2)

class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
    def forward(self,x):
        return x*torch.sigmoid(x)

def init_weights(m):
    def truncated_normal_init(t,mean = 0.0,std = 0.01):
        torch.nn.init.normal_(t,mean=mean,std=std)
        while True:
            cond = (t < mean - 2*std) | (t > mean + 2*std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond,
                torch.nn.init.normal_(torch.ones(t.shape),mean = mean,std = std),t

            )
        return t
    
    if type(m) == nn.Linear or isinstance(m,FCLayer):
        truncated_normal_init(m.weight,std = 1/(2*np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0)
    

class FCLayer(nn.Module):
    def __init__(self,input_dim,output_dim,ensemble_size,activation):
        super(FCLayer,self).__init__()
        self._input_dim,self._output_dim = input_dim,output_dim
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size,input_dim,output_dim)
        )
        self._activation = activation
        self.bias = nn.Parameter(torch.Tensor(ensemble_size,output_dim))

    def forward(self,x):
        return self._activation(torch.add(torch.bmm(x,self.weight)))


# Ensemble Model

class EnsembleModel(nn.Module):
    def __init__(self,state_dim,action_dim,model_alpha,ensemble_size = 5, learning_rate = 1e-3):
        super(EnsembleModel,self).__init__()
        self._output_dim = (state_dim + 1) * 2 # As we need to output var and mean
        self._model_alpha = model_alpha
        self._max_logvar = nn.Parameter((torch.ones((1,self._output_dim//2)).flaot()/2).to(device),requires_grad=False)
        self._min_logvar = nn.Parameter((-torch.ones((1,self._output_dim//2)).float()*10).to(device),requires_grad=False)
        self.layer1 = FCLayer(200,200,ensemble_size,Swish())
        self.layer2 = FCLayer(200,200,ensemble_size,Swish())
        self.layer3 = FCLayer(200,200,ensemble_size,Swish())
        self.layer4 = FCLayer(200,200,ensemble_size,Swish())
        self.layer5 = FCLayer(200,self._output_dim,ensemble_size,nn.Identity())
        self.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(),lr = learning_rate)

    def forward(self,x,return_log_var = False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        mean = ret[:,:,:self._output_dim//2]
        # Control the var between max and min
        logvar = self._max_logvar - F.softplus(self._max_logvar - ret[:,:,self._output_dim//2:])
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar if return_log_var else torch.exp(logvar)
    
    def loss(self,mean,logvar,labels,use_var_loss = True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels,2)*inverse_var,dim = -1),dim = -1)
            var_loss = torch.mean(torch.mean(logvar,dim = -1),dim = -1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        
        else:
            mse_loss = torch.mean(torch.pow(mean-labels,2),dim = (1,2))
            total_loss = torch.sum(mse_loss)
        return total_loss,mse_loss
    
    def train(self,loss):
        self.optimizer.zero_grad()
        loss += self._model_alpha * torch.sum(self._max_logvar) - self._model_alpha * torch.sum(self._min_logvar)
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    def __init__(self,state_dim,action_dim,model_alpha = 0.01, num_network = 5):
        self._num_network = num_network
        self._state_dim,self._action_dim = state_dim,action_dim
        self.model = EnsembleModel(state_dim,action_dim,model_alpha,ensemble_size=num_network)
        self._epoch_since_last_update = 0

    def train(self,inputs,labels,batch_size = 64,holdout_ratio = 0.1,max_iter = 20):
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation],labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:],labels[num_holdout:]
        holdout_inputs,holdout_labels = inputs[:num_holdout],labels[:num_holdout]
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None,:,:].repeat([self._num_network,1,1])
        holdout_labels = holdout_labels[None,:,:].repeat([self._num_network,1,1])
        self._snapshots = {i: (None,1e10) for i in range(self._num_network)}

        for epoch in itertools.count():
            train_index = np.vstack([
                np.random.permutation(train_inputs.shape[0])
                for _ in range(self._num_network)
            ])

            for batch_start_pos in range(0,train_inputs.shape[0],batch_size):
                batch_index = train_index[:,batch_start_pos:batch_start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[batch_index]).float().to(device)
                train_label = torch.from_numpy(train_labels[batch_index]).float().to(device)
                mean,logvar = self.model(train_input,return_log_var = True)
                loss, _  = self.model.loss(mean,logvar,train_label)
                self.model.train(loss)

    def _save_best(self,epoch,losses,threshold = 0.1):
        updated = False
        for i in range(len(losses)):
            current = losses[i]
            _, best = self._snapshots[i]
            improvements = (best - current) / best
            if improvements > threshold:
                self._snapshots[i] = (epoch, current)
                updated = True
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5
    
    def predict(self,inputs,batch_size = 64):
        inputs = np.tile(inputs,(self._num_network,1,1))
        inputs = torch.tensor(inputs,dtype = torch.float).to(device)
        mean,var = self.model(inputs,return_log_var = False)
        return mean.detach().cpu().numpy() , var.detach().cpu().numpy()



class FakeEnv:
    def __init__(self,model):
        self.model = model
    
    def step(self,obs,act):
        inputs = np.concatenate((obs,act),axis = -1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_samples = ensemble_model_means + np.random.normal(size = ensemble_model_means.shape) * ensemble_model_stds
        num_models, batch_size, _ = ensemble_model_means.shape

        models_to_use = np.random.choice([i for i in range(self.model._num_network)],size = batch_size)
        batch_inds = np.arange(0,batch_size)
        samples = ensemble_samples[models_to_use,batch_inds]
        rewards,next_obs = samples[:,:1][0][0],samples[:,1:][0]
        return rewards,next_obs
    
class MBPO:
    def __init__(self):
        pass
    




