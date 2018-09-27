import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


with open (os.path.join('./pkl','lander_27-09-2018_10-21-54.pkl'), 'rb') as fo:
    data = pickle.load(fo)
# print(data)
episode_rewards, t, mean_episode_rewards, best_mean_episode_rewards = data['reward'], data['timestep'], data['mean_reward'], data['best_reward']
# Assign first value
# best_mean_episode_rewards[0] = mean_episode_rewards[0]
if 0:
    print(np.shape(episode_rewards))
    print(np.shape(t))
    print(np.shape(mean_episode_rewards))
    print(np.shape(best_mean_episode_rewards))
    print(best_mean_episode_rewards)

# Plot result
sns.set(style="darkgrid", font_scale=1.5)
sns.tsplot(time=t, data=mean_episode_rewards, color='k', linestyle='-.')
sns.tsplot(time=t, data=best_mean_episode_rewards, color='r', linestyle='-')
plt.legend(loc='best').draggable()
plt.show()

