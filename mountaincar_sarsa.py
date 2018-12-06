import gym
import numpy as np
import matplotlib.pyplot as plt

#exploring Mountain Car environment

env = gym.make('MountainCar-v0')
# assign the hyperparameters

n_states = 40 # number of states
episodes = 20 # number of episodes/runs
initial_lr = 1.0 # number of learning rate
min_lr = 0.005 # minimum learning rate
gamma = 0.99 # discount factor
max_steps = 300
epsilon = 0.05
episodescores = []

iterations = np.zeros([episodes, episodes], int)
average = np.zeros(episodes, float)
std_deviation = np.zeros(episodes, float)

env = env.unwrapped
env.seed(0) #setting environment seed to reproducde same result
np.random.seed(0) #setting numpy number generation seed to reproduce same random numbers


# perform discretization of the continuous state
# descretization is the conversion of continuous states space observation

def discretization (env, obs):
    env_low = env.observation_space.low;
    env_high = env.observation_space.high;

    env_den = (env_high - env_low) / n_states
    pos_den = env_den[0]
    vel_den = env_den[1]

    pos_high = env_high[0]
    pos_low  = env_low[0]
    vel_high = env_high[1]
    vel_low  = env_low[1]

    pos_scaled = int((obs[0] - pos_low)/pos_den)
    vel_scaled = int((obs[1] - vel_low)/vel_den)

    return pos_scaled, vel_scaled

# Q-learning algorithm by initializing a Q-table/ updating the Q-values

# Q-table
# rows are states but her state is 2-D pos, vel
# columns are actions 
# Q-table would be 3-D

q_table = np.zeros((n_states, n_states, env.action_space.n))
total_steps = 0


for episode in range(episodes):
    for i_episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        #decreasing learning rate alpha over time
        alpha = max(min_lr, initial_lr*(gamma **(episode//100)))
        steps = 0
        #action for the current state using epsilon greedy
        if np.random.uniform(low = 0, high = 1) < epsilon :
            act = np.random.choice(env.action_space.n)
        else :
            pos, vel = discretization(env, obs)
            act = np.argmax(q_table[pos][vel])

        # for t in range(10000):
        while True:
            # env.render()
            pos,vel = discretization(env, obs)

            pos_,vel_ = discretization(env, obs)
            policy = np.ones(env.action_space.n) * epsilon / env.action_space.n
            act = np.argmax(q_table[pos][vel])
            policy[act] += 1. - epsilon

            act_ = np.random.choice(env.action_space.n, p=policy)

            # if np.random.uniform(low = 0, high = 1) < epsilon:
            # 	act = np.random.choice(env.action_space.n)
            # else : 
            # 	act = np.argmax(q_table[pos][vel])

            obs, reward, done, _ = env.step(act_)
            # total_reward += abs(obs[0] + 0.5)
            total_reward += reward


            # if t > 0:
            q_table[pos][vel][act] = q_table[pos][vel][act] + alpha *(reward + gamma * q_table[pos_][vel_][act_] - q_table[pos][vel][act])
            pos_,vel_ = discretization(env, obs)
            act = act_
            steps += 1
            if done :
                # env.close()
                break
        episodescores.append(total_reward)
        iterations[episode][i_episode] = steps + 1
        average[episode] += iterations[episode][[i_episode]] *1.0 / episodes
    std_deviation = np.std(iterations[episode], ddof = 1)
    print("Episode {} completed with total reward {} in {} steps".format(episode+1, total_reward, steps))

# print("Average ", np.mean(episodescores))
# while True:  #to hold the render at the last step when Car passes the flag
#   env.render()
#   plot_running_avg(total_reward)
plt.title('MountainCar: sarsa')
# plt.plot(episodescores)
# plt.xlabel('Eplside')
# plt.ylabel('Reward of Episode')
x = range(episodes)  
plt.xticks(np.arange(0, 21, 1.0))
plt.errorbar(x, average, yerr=std_deviation, fmt='-o')
plt.figtext(0.5, 0.01, "Mean value is {} and mean of reward is {}".format(np.mean(average), np.mean(episodescores)), fontsize = 8, va = "bottom", ha = "center")
plt.show()




