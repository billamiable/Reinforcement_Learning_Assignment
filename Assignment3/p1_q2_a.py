import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Basic Q Learning
with open (os.path.join('./pkl','ZaxxonNoFrameskip-v4_02-10-2018_05-08-06.pkl'), 'rb') as fb:
    bas = pickle.load(fb)
bas_r, bas_t, bas_mean_r, bas_best_mean_r = bas['reward'], bas['timestep'], bas['mean_reward'], bas['best_reward']
# Double Q Learning
with open (os.path.join('./pkl','AsterixNoFrameskip-v4_03-10-2018_01-23-19.pkl'), 'rb') as fd:
    db = pickle.load(fd)
db_r,  db_t,  db_mean_r,  db_best_mean_r  = db['reward'],  db['timestep'],  db['mean_reward'],  db['best_reward']

# Plot result
sns.set(style="darkgrid", font_scale=1.5)
f, ax = plt.subplots(1, 1)
ax.plot(bas_t, bas_mean_r,      color="red",  label="mean_vanilla", linestyle="-")
ax.plot(bas_t, bas_best_mean_r, color="red",  label="best_vanilla", linestyle="-.")
ax.plot(db_t,  db_mean_r,       color="blue", label="mean_double",  linestyle="-")
ax.plot(db_t,  db_best_mean_r,  color="blue", label="best_double",  linestyle="-.")
ax.legend()

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel("Episode Rewards", fontsize=15)
plt.xlabel("Number of Timesteps", fontsize=15, labelpad=4)
plt.title ("Vanilla and Double Q Learning Performance for Zaxxon", fontsize=20)
plt.legend(loc='lower right')
plt.show()

