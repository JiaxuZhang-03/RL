import numpy as np
from temporal_difference import *

class nstep_Sarsa:
    def __init__(self,n,ncol,nrow,epsilon,alpha,gamma,n_action = 4):
        self.Q_table = np.zeros([ncol * nrow,n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.state_list = []
        self.action_list = []
        self.reward_list = []
    
    def take_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self,state):
        Q_max = np.argmax(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state,i] == Q_max:
                a[i] = 1
        return a
    
    def update(self,s0,a0,r,s1,a1,done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:
            G = self.Q_table[s1,a1]
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]

                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s,a] += self.alpha *(G - self.Q_table[s,a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s,a] += self.alpha * (G - self.Q_table[s,a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []


np.random.seed(0)
n_step = 5
epsilon = 0.1
alpha = 0.1
env = CliffWalkingEnv(12,4)
gamma = 0.9
agent = nstep_Sarsa(n_step,12,4,epsilon,alpha,gamma)
num_episodes = 500

return_list = []

for i in range(10):  
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward  
                agent.update(state, action, reward, next_state, next_action,
                             done)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
plt.show()
        