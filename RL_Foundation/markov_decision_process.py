import numpy as np
np.random.seed(0)

P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

P = np.array(P)
rewards = [-1,-2,-2,10,1,0]
gamma = 0.5

def compute_return(start_index,chain,gamma):
    G = 0
    for i in reversed(range(start_index,len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

chain = [1,2,3,6]
start_index = 0
G = compute_return(start_index,chain,gamma)
print("Total Reward is:%s"%G)

# Bellman Equation Solver
def compute(P,rewards,gamma,states_num):
    rewards = np.array(rewards).reshape(-1,1)
    value = np.dot(np.linalg.inv(np.eye(states_num,states_num)-gamma*P),rewards)
    return value

V = compute(P,rewards,gamma,6)
print("Every value in the MRP state is \n",V)


def sample(MDP,Pi,timestep_max,number):
    S,A,P,R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]
        while s != "s5" and timestep < timestep_max:
            timestep += 1
            rand,temp = np.random.rand(),0
            for a_opt in A:
                temp += Pi.get((s,a_opt),0)
                if temp > rand:
                    a = a_opt
                    r = R.get((s,a),0)
            rand,temp = np.random.rand(),0
            for s_opt in S:
                temp += P.get((s,a,s_opt),0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s,a,r,s_next))
            s = s_next
        episodes.append(episode)
    return episodes


def MC(episodes,V,N,gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1,-1,-1):
            (s,a,r,s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G-V[s])/N[s]
    
def occupancy(episodes,s,a,timestep_max,gamma):
    rho = 0
    total_times = np.zeros(timestep_max)
    occur_times = np.zeros(timestep_max)
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt,a_opt,r,s_next) = episode[i]
            total_times[i] += 1
            if s==s_opt and a==a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i]/total_times[i] # Using Frequency as Prob
    return (1-gamma)*rho