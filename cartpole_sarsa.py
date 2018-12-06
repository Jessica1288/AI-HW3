import random
import math
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
env = env.unwrapped
env.seed(0) #setting environment seed to reproducde same result
np.random.seed(0) #setting numpy number generation seed to reproduce same random numbers

# outdir = sys.argv[2]

initial_epsilon = 0.1 # probability of choosing a random action (changed from original value of 0.0)
alpha = 0.05 # learning rate
lambda_ = 0.9 # trace decay rate
gamma = 0.99 # discount rate
N = 3000 # memory for storing parameters 
episodes = 20
# M = env.action_space.n
NUM_TILINGS = 10
NUM_TILES = 8
episodescores = []
iterations = np.zeros([episodes, episodes], int)
average = np.zeros(episodes, float)
std_deviation = np.zeros(episodes, float)

def main():
    # env.monitor.start(outdir)

    epsilon = initial_epsilon
    theta = np.zeros(N) # parameters (memory)

    for episode_num in range(episodes):
        for i_episode_num in range(episodes):
            episodescore, iteration = episode(episode_num, i_episode_num, epsilon, theta, env.spec.timestep_limit)
            epsilon = epsilon * 0.999 # added epsilon decay
            episodescores.append(episodescore)
        std_deviation = np.std(iteration, ddof = 1)

    x = range(episodes)   
    plt.title('CartPole-v1: Sarsa')
    # plt.plot(episodescores)
    # plt.xlabel('Eplside')
    # plt.ylabel('Reward of Episode')
    # plt.xticks(np.arrange(0, x, 1.0))
    plt.xticks(np.arange(0, 21, 1.0))
    plt.errorbar(x, average, yerr=std_deviation, fmt='-o')
    plt.figtext(0.5, 0.01, "Mean value is {} and mean of reward is {}".format(np.mean(average), np.mean(episodescores)), fontsize = 8, va = "bottom", ha = "center")
    plt.show()
    # env.monitor.close()

def episode(episode_num, i_episode_num, epsilon, theta, max_steps):
    Q = np.zeros(env.action_space.n) # action values
    e = np.zeros(N) # eligibility traces
    F = np.zeros((env.action_space.n, NUM_TILINGS), dtype=np.int32) # features for each action
    total_reward = 0
    def load_F(observation):
        state_vars = []
        for i, var in enumerate(observation):
            range_ = (env.observation_space.high[i] - env.observation_space.low[i])
            # in CartPole, there is no range on the velocities, so default to 1
            if range_ == float('inf'):
                range_ = 1
            state_vars.append(var / range_ * NUM_TILES)

        for a in range(env.action_space.n):
            F[a] = get_tiles(NUM_TILINGS, state_vars, N, a)

    def load_Q():
        for a in range(env.action_space.n):
            Q[a] = 0
            for j in range(NUM_TILINGS):
                Q[a] += theta[F[a,j]]

    observation = env.reset()
    load_F(observation)
    load_Q()
    action = np.argmax(Q) # numpy argmax chooses first in a tie, not random like original implementation
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    # else :
    # 	action = np.argmax()
    step = 0
    while True:
        # env.render()
        step += 1
        e *= gamma * lambda_
        for a in range(env.action_space.n):
            v = 0.0
            if a == action:
                v = 1.0

            for j in range(NUM_TILINGS):
                e[F[a,j]] = v

        observation, reward, done, info = env.step(action)
        delta = reward + Q[action]
        load_F(observation)
        load_Q()
        # next_action = np.argmax(Q)
        policy = np.ones(env.action_space.n) * epsilon / env.action_space.n
        act = np.argmax(Q[action])
        policy[act] += 1. - epsilon
        next_action = np.random.choice(env.action_space.n, p=policy)
        if np.random.random() < epsilon:
            next_action = env.action_space.sample()
        if not done:
            delta = Q[action] + alpha * (reward + gamma * Q[next_action] - Q[action])
        theta += delta
        load_Q()
        total_reward += reward
        if done or step > max_steps:
            # env.close()
            break
        action = next_action
    iterations[episode_num][i_episode_num] = step + 1
    average[episode_num] += iterations[episode_num][[i_episode_num]]*1.0 / episodes
    print("Episode {} completed with total reward {} in {} steps".format(episode_num + 1, total_reward, step))
    return total_reward, iterations[episode_num]


def get_tiles(num_tilings, variables, memory_size, hash_value):
    num_coordinates = len(variables) + 2
    coordinates = [0 for i in range(num_coordinates)]
    coordinates[-1] = hash_value

    qstate = [0 for i in range(len(variables))]
    base = [0 for i in range(len(variables))]
    tiles = [0 for i in range(num_tilings)]

    for i, variable in enumerate(variables):
        qstate[i] = int(math.floor(variable * num_tilings))
        base[i] = 0

    for j in range(num_tilings):
        for i in range(len(variables)):
            if (qstate[i] >= base[i]):
                coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings)
            else:
                coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % num_tilings) - num_tilings

            base[i] += 1 + (2 * i)
        coordinates[len(variables)] = j
        tiles[j] = hash_coordinates(coordinates, memory_size)

    return tiles

rndseq = np.random.randint(0, 2**32-1, 2048)

def hash_coordinates(coordinates, memory_size):
    total = 0
    for i, coordinate in enumerate(coordinates):
        index = coordinate
        index += (449 * i)
        index %= 2048
        while index < 0:
            index += 2048

        total += rndseq[index]

    index = total % memory_size
    while index < 0:
        index += memory_size

    return index

main()