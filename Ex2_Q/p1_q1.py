import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Basic Q Learning
with open (os.path.join('./pkl','lander_05-12-2018_04-19-36_0.01.pkl'), 'rb') as fo:
    bas = pickle.load(fo)
bas_r, bas_t, bas_mean_r, bas_best_mean_r = bas['reward'], bas['timestep'], bas['mean_reward'], bas['best_reward']

# Plot result
sns.set(style="darkgrid", font_scale=1.5)
f, ax = plt.subplots(1, 1)
ax.plot(bas_t, bas_mean_r,      color="red",  label="mean_100-episode_reward", linestyle="-")
ax.plot(bas_t, bas_best_mean_r, color="red",  label="best_mean_reward", linestyle="-.")
ax.legend()

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel("Episode Rewards", fontsize=15)
plt.xlabel("Number of Timesteps", fontsize=15, labelpad=4)
plt.title ("Soft Q Learning with EX2 Exploration Performance for Lander", fontsize=20)
plt.legend(loc='lower right')
plt.show()