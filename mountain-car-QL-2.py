import gym
from matplotlib import pyplot as plt
import numpy as np

DEBUG = True
PLOT = False
RENDER = True
NUM_EPISODES = 5000
EPSILON_S= 0.3
# discount = 0.9
# learning_rate = 0.1
SCALE = np.array([10, 100]) # observation space - cont -> discrete

#initialize the exploration probability to 1
# epsilon = 1

#exploartion decreasing decay for exponential decreasing
epsilon_decay = 0.001

# minimum of exploration proba
min_epsilon = 0.01

def q_table_init():
  # Initialize the Q-table with random probabilities (0-1)
  # 19 (discrete observation space- position) 
  # x 15 (discrete observation space- velocity) 
  # x 3 (action space- left, none, right - acceleration) 
  return np.random.uniform(size = (19, 15, 3))

def prepro(state):
  """ prepro continuous Box([-1.2 -0.07], [0.6 0.07], (2,), float32) into discrete 285 (19x15) 1D float vector """
  # print(state)
  state_delta_scaled = (state - np.array([-1.2, -0.07]))*SCALE
  # print(state_delta_scaled)
  return np.round(state_delta_scaled, 0).astype(int)


# def get_action_epsilon_greedy(state, epsilon):
#   if np.random.random() < 1 - epsilon:
#       return np.argmax(Q_table[state[0], state[1]]) 
#   else:
#       return np.random.randint(0, 3)
   
# def reset_env():
#   return env.reset(), 0, False

def q_learn(learning_rate, discount, epsilon):
    # Mountain Car env
    env = gym.make('MountainCar-v0')
    Q_table = q_table_init()

    reward_s = [] # rewards received for taking an action corresponding to input state/time step
    reward_s_avgs = []

    episode_count = 0

    while episode_count <= NUM_EPISODES:
        current_state = env.reset()
        current_state = prepro(current_state)
        terminated = False
        
        #sum the rewards that the agent gets from the environment
        total_episode_reward = 0

        while not terminated:
            if RENDER and NUM_EPISODES - episode_count < 20: env.render()
        
            if np.random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[current_state[0], current_state[1],:])

            next_state, reward, terminated, _ = env.step(action)
            next_state = prepro(next_state)

            p1 = (1-learning_rate) * Q_table[current_state[0], current_state[1],  action]
            p2 = learning_rate*(reward + discount*max(Q_table[next_state[0], next_state[1], :]))
            Q_table[current_state[0], current_state[1], action] = p1 + p2
            total_episode_reward = total_episode_reward + reward

            current_state = next_state

        epsilon = max(min_epsilon, np.exp(-epsilon_decay*episode_count))
        reward_s.append(total_episode_reward)
        reward_s_avgs.append(np.mean(reward_s))

        if DEBUG and (episode_count+1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(episode_count+1, np.mean(reward_s)))

        episode_count += 1

    env.close()
    return reward_s_avgs


if __name__ == '__main__':
    q_learn(0.1, 0.999, 1)

    if PLOT:
        params = {
            'learning_rates': [0.1, 0.01, 0.001],
            'discount_rates': [0.9, 0.99, 0.999],
            'epsilon_values': [1, .1, .5],
        }
        title = "Average Reward vs Episodes"
        xlabel = 'Episodes'
        ylabel = 'Average Reward'

        for p in params:
            if p == 'epsilon_values':
                print("Writing", f'rewards-{p}.jpg')

                plt.suptitle(title, fontsize=18)
                title = ''
                for p2 in params:
                    if p != p2:
                        title += f"{p2}: {params[p2][0]}; "
                plt.title(title, fontsize=10)

                for v in params[p]:
                    if p == 'learning_rates':
                        reward_s_avgs = q_learn(v, params['discount_rates'][0], params['epsilon_values'][0])
                    elif p == 'discount_rates':
                        reward_s_avgs = q_learn(params['learning_rates'][0], v, params['epsilon_values'][0])
                    elif p == 'epsilon_values':
                        reward_s_avgs = q_learn(params['learning_rates'][0], params['discount_rates'][0], v)
                    else:
                        print("Error")
                        quit()

                    plt.plot((np.arange(len(reward_s_avgs)) + 1), reward_s_avgs)
                plt.legend(params[p])

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.savefig(f'rewards-{p}.jpg')     
                plt.close()  