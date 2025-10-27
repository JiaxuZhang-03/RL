import gym
from RL_Foundation.dp_algo import *

# --- Env --- 

env = gym.make("FrozenLake-v1")
env = env.unwrapped
env.render()

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])

holes = holes - ends
print("Index of holes:", holes)
print("Index of Targets:",ends)

action_meaning = ["<","v",">","^"]
theta = 1e-5
gamma = 0.9
# agent = PolicyIteration(env,theta,gamma)
# agent.policy_evaluation()
# print_agent(agent,action_meaning,[5,7,11,12],[15])


agent = ValueIteration(env,theta,gamma)
agent.value_iteration()
print_agent(agent,action_meaning,disaster = [5,7,11,12],end = [15])

